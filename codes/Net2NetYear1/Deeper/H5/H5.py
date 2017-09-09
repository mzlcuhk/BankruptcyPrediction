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
        epochs_drop = 500.0
        lrate = initial_lrate * math.pow(drop,((1+epoch)/epochs_drop))
        return lrate





#Load Teacher_Model
teacher_model = load_model('H4_teacher.h5')

# extract Teacher_Model Weights
W1b1 = teacher_model.get_layer('h1').get_weights()
W2b2 = teacher_model.get_layer('h2').get_weights()
W3b3 = teacher_model.get_layer('h3').get_weights()
W4b4 = teacher_model.get_layer('h4').get_weights()
Wobo = teacher_model.get_layer('out').get_weights()

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



print "DataRead"

# Data Split
X_train, X_test, y_train, y_test = data_split(X, Y, test_size=0.2, random_state=seed)
#X_train, X_val, y_train, y_val = data_split(X_train1, y_train1, test_size=0.1, random_state=seed)
y_train = to_categoricalUD(y_train)
y_val = to_categoricalUD(y_val)
y_test = to_categoricalUD(y_test)

#X_val = numpy.concatenate((X_test,X_val),axis=0)
#y_val = numpy.concatenate((y_test,y_val),axis=0)

#for i in range(X.shape[1]):
#        X_train[:,i]=numpy.subtract(X_train[:,i],X_train[:,i].mean())
#        X_val[:,i]=numpy.subtract(X_val[:,i],X_val[:,i].mean())
#        X_test[:,i]=numpy.subtract(X_test[:,i],X_test[:,i].mean())


# create model
model = Sequential()
prelu=keras.layers.advanced_activations.PReLU()
prelu1=keras.layers.advanced_activations.PReLU()
prelu2=keras.layers.advanced_activations.PReLU()
prelu3=keras.layers.advanced_activations.PReLU()
prelu4=keras.layers.advanced_activations.PReLU()
HiddenNeurons = 128


model.add(Dense(HiddenNeurons, input_dim=X_train.shape[1], init='uniform',name='h1'))#, activation='relu'))
model.add(prelu)
model.add(Dropout(0.2))

model.add(Dense(HiddenNeurons, name='h2'))#, activation='relu'))
model.add(prelu1)
model.add(Dropout(0.2))

model.add(Dense(HiddenNeurons, init='identity',name='h3'))#, activation='relu'))
model.add(prelu2)
model.add(Dropout(0.5))

model.add(Dense(HiddenNeurons, init='identity',name='h4'))#, activation='relu'))
model.add(prelu3)
model.add(Dropout(0.5))

model.add(Dense(HiddenNeurons, init='identity',name='h5'))#, activation='relu'))
model.add(prelu4)
model.add(Dropout(0.45))

model.add(Dense(2, init='uniform',activation='softmax',name='out'))
#model.add(Activation('softmax'))




# Set the calculated weights and biases
model.get_layer('h1').set_weights(W1b1)
model.get_layer('h2').set_weights(W2b2)
model.get_layer('h3').set_weights(W3b3)
model.get_layer('h4').set_weights(W4b4)
model.get_layer('out').set_weights(Wobo)



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
model.fit(X_train, y_train,validation_data=(X_test,y_test),nb_epoch=2000, batch_size=X_train.shape[0], callbacks=callbacks_list, verbose=1)
#model.fit(X_train, y_train,validation_data=(X_test,y_test),nb_epoch=550, batch_size=X_train.shape[0],class_weight={0:1, 1:6756.0/271}, callbacks=callbacks_list, verbose=1)

#Model prediction
predicted=model.predict_proba(X_test,batch_size=25)
predicted1=model.predict_proba(X_val,batch_size=25)
pred=model.predict_classes(X_test,batch_size=25)

y_val = numpy.argmax(y_val,1)
yt = numpy.argmax(y_test,1)
print "\n\nROC_AUC: ", roc_auc_score(yt, predicted[:,1])
print "\n\nROC_AUC Val Data: ", roc_auc_score(y_val, predicted1[:,1])

numpy.save("Prediction.npy",predicted)
numpy.save("Xtest.npy",X_test)
model.save('H5_student.h5')

