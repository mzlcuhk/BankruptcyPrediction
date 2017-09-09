The "Net2Net" folder contains the codes to implement Net2Net algorithm proposed in the paper by Goodfellow et. al. [1]. The basic idea is that we add neurons in steps. Each new neuron (STUDENT NEURON) is initiallized by the weights of the previously trained neurons (TEACHER NEURON). By doing so, it each neuron ends up learning a different function than the previous neuron. And thus accelerating the learning by knowledge transfer. 

Implementation: 

The folder "Wider" has neurons of the 1st hidden layer. The weights of N1 are saved as N1_student.h5 and copied to N2 as N1_teacher.h5. The weights of N2 are then stored as N2_student.h5 and copied to N4 as N2_teacher.h5 and so on. Finally weights of N128 are transfered to folder "Deeper" (so in this case Hidden1 has 128 neurons) in folder "H2" as H1_teacher.h5. They are then used to initialize H1 and neurons of H2 are initialized with "identity" initialization. 



References:

[1] Chen, Tianqi, Ian Goodfellow, and Jonathon Shlens. "Net2net: Accelerating learning via knowledge transfer." arXiv preprint arXiv:1511.05641 (2015).

