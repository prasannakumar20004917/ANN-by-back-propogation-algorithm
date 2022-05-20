### EX NO : 6
### DATE  :29.04.2022
# <p align="center"> ANN BY BACK PROPAGATION ALGORITHM </p>
## Aim:
   To implement multi layer artificial neural network using back propagation algorithm.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner /Google Colab

## Related Theory Concept:
Algorithm for ANN Backpropagation: • Weight initialization: Set all weights and node thresholds to small random numbers. Note that the node threshold is the negative of the weight from the bias unit(whose activation level is fixed at 1).

• Calculation of Activation:

1.The activation level of an input is determined by the instance presented to the network.
2.The activation level oj of a hidden and output unit is determined.
• Weight training:

1.Start at the output units and work backward to the hidden layer recursively and adjust weights.

2.The weight change is completed.

3.The error gradient is given by:

a. For the output units.

b. For the hidden units.

Repeat iterations until convergence in term of the selected error criterion. An iteration includes presenting an instance, calculating activation and modifying weights.


## Algorithm
1.Import packages

2.Defining Sigmoid Function for output

3.Derivative of Sigmoid Function

4.Initialize variables for training iterations and learning rate

5.Defining weight and biases for hidden and output layer

6.Updating Weights

## Program:
```
/*
Program to implement ANN by back propagation algorithm.
Developed by   :Prasannakumar M
RegisterNumber : 212220230035 
*/
```
```python
import numpy as np
X=np.array(([2,9],[1,5],[3,6]), dtype=float)
y=np.array(([92],[86],[89]), dtype=float)
X=X/np.amax(X,axis=0) #maximum of X array longitudinally
y=y/100
#Sigmoid function
def sigmoid(x):
  return 1/(1+np.exp(-x))
#Derivative of sigmoid function
def derivatives_sigmoid(x):
  return x * (1-x)
#variable initialization
epoch=7000 #setting training iterations
lr=0.1 #setting learning rates
inputlayer_neurons = 2 #no. of features in dataset
hiddenlayer_neurons = 3 #no. of hidden layer neurons
output_neurons = 1 #number of neurons at output layer
#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
#draws a random range of numbers uniformly of dim x*y
for i in range(epoch):
  hinp1=np.dot(X,wh)
hinp=hinp1 + bh
hlayer_act = sigmoid(hinp)
outinp1=np.dot(hlayer_act,wout)
outinp= outinp1+ bout
output = sigmoid(outinp)
#Backpropagation
EO = y-output
outgrad = derivatives_sigmoid(output)
d_output = EO* outgrad
EH = d_output.dot(wout.T)
hiddengrad = derivatives_sigmoid(hlayer_act)
d_hiddenlayer = EH*hiddengrad
wout+=hlayer_act.T.dot(d_output) *lr #dot product of nextlayer error and current
#bout += np.sum(d_output, axis=0,keepdims=True) *lr
wh += X.T.dot(d_hiddenlayer) *lr
#bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) *lr
print("Input: \n" + str(X))
print("Actual output: \n" + str(y))
print("Predicted output: \n" , output)
```

## Output:
![exp66](https://user-images.githubusercontent.com/75235090/169484749-4504b61e-fe98-4111-8ce7-3cd65e175bac.png)



## Result:
Thus the python program successully implemented multi layer artificial neural network using back propagation algorithm.
