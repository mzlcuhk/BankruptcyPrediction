import numpy,math
import keras,csv,os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.utils import check_array
from keras.utils.visualize_util import plot
from keras.callbacks import LearningRateScheduler
from keras.callbacks import History

from keras.layers import Activation
from keras.utils.np_utils import to_categorical
#from sklearn.metrics import accuracy_score
from keras import metrics
from keras import backend as K
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def to_categoricalUD(y):
	yc = numpy.zeros((y.shape[0],2))
	j = 0
	for i in range(y.shape[0]):
		if y[i] == 0:
			yc[i,0] = 1
			j += 1
		elif y[i] == 1:
			yc[i,1] = 1
			j += 1
		else:
			yc[i,2] = 1
			j += 1

	return yc


def data_split(X,y,n_splits=3, test_size=0.2, random_state=0):
	sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
	sss.get_n_splits(X, y)
	for train_index, test_index in sss.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

	return X_train, X_test, y_train, y_test


def acc(t,l):
	tru = numpy.argmax(t,1)
	pred = numpy.argmax(l,1)
	acc = numpy.mean(tru == pred)
	return acc

# learning rate schedule
def step_decay(epoch):
        initial_lrate = 0.005
        drop = 0.5
        epochs_drop = 400.0
        lrate = initial_lrate * math.pow(drop,((1+epoch)/epochs_drop))
        return lrate




#Load Teacher_Model
teacher_model = load_model('N32_teacher.h5')

# extract Teacher_Model Weights
W1b1 = teacher_model.get_layer('h1').get_weights()
W2b2 = teacher_model.get_layer('out').get_weights()

teacher_W1 = W1b1[0]
teacher_b1 = W1b1[1]
teacher_W2 = W2b2[0]
teacher_b2 = W2b2[1]


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
X_val = numpy.load('../../../Data/SmoteYear1DataTest.npy')
y_val = numpy.load('../../../Data/SmoteYear1labelTest.npy')
X = numpy.load('../../../Data/SmoteYear1Data.npy')
Y = numpy.load('../../../Data/SmoteYear1label.npy')

# Data Split
X_train, X_test, y_train, y_test = data_split(X, Y, test_size=0.2, random_state=seed)
#X_train, X_val, y_train, y_val = data_split(X_train1, y_train1, test_size=0.1, random_state=seed)
y_train = to_categoricalUD(y_train)
y_val = to_categoricalUD(y_val)
y_test = to_categoricalUD(y_test)



# create model
model = Sequential()
prelu=keras.layers.advanced_activations.PReLU()
HiddenNeurons = 64
TeacherNeurons = 32

model.add(Dense(HiddenNeurons, input_dim=X_train.shape[1], init='uniform',name='h1'))#, activation='relu'))
model.add(prelu)

model.add(Dense(2, init='uniform',activation='softmax',name='out'))
#model.add(Activation('softmax'))

########################## Widen operation ##########################
ExtraNeurons = HiddenNeurons-TeacherNeurons
# Randon Idx selection for Neuron Replication
idx = numpy.random.randint(teacher_W1.shape[1],size=ExtraNeurons)
# Replicate input weights of new neurons with weights of neurons in corresponding indices
tmpW1 = teacher_W1[:,idx]
# Add the new neurons to the weight matrix
student_W1 = numpy.concatenate((teacher_W1,tmpW1),axis=1)
# Replicate biases of new neurons with those of neurons in corresponding indices
tmpb1 = teacher_b1[idx]
# Add new neurons to bias vector
student_b1 = numpy.concatenate((teacher_b1,tmpb1))
# Take count of number of neurons of the same type being replicated
scaler = numpy.bincount(idx)[idx] + 1 #******** +1 for already existing neuron
# Scale down output weights according of new neurons by the scaler
tmpW2 = teacher_W2[idx,:]/scaler[:,None]
# Add some white noise to the new weights to enable faster training
noisyW2 = tmpW2+numpy.random.normal(0,1e-4,size=tmpW2.shape)
# Add new noisy neurons to output weight matrix
student_W2 = numpy.concatenate((teacher_W2,noisyW2),axis=0)
# Scale down existing neurons that were replicated.
student_W2[idx,:] = tmpW2
# Equate biases of next layer neurons to that of teacher network
student_b2 = teacher_b2 #******************* No change in output weights

# Set the calculated weights and biases
model.get_layer('h1').set_weights([student_W1, student_b1])
model.get_layer('out').set_weights([student_W2, student_b2])


# Compile model
current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, "FullyConnectedNetworkPrelu.png")
plot(model, to_file=model_path, show_shapes=True)
adam=keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy', optimizer=adam,metrics=['categorical_accuracy'])

# learning schedule callback
history=History()
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate,history]

#model Fitting
print "Training..."
model.fit(X_train, y_train,validation_data=(X_test,y_test),nb_epoch=2500, batch_size=X_train.shape[0], callbacks=callbacks_list, verbose=1)
#model.fit(X_train, y_train,validation_data=(X_test,y_test),nb_epoch=550, batch_size=X_train.shape[0],class_weight={0:1, 1:6756.0/271}, callbacks=callbacks_list, verbose=1)

#Model prediction
predicted=model.predict_proba(X_test,batch_size=25)
predicted1=model.predict_proba(X_val,batch_size=25)

y_val = numpy.argmax(y_val,1)

print "\n\nROC_AUC Val Data: ", roc_auc_score(y_val, predicted1[:,1])

numpy.save("Prediction.npy",predicted)
numpy.save("Xtest.npy",X_test)
model.save('N64_student.h5')

