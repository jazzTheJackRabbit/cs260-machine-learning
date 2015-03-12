from Classifier import Classifier
import numpy as np
import math
import random
import string

class NeuralNetwork(Classifier):
    def __init__(self, numberOfInputNeurons, numberOfHiddenNeurons, numberOfOutputNeurons):        
        #Include the BIAS node in input
        self.numberOfInputNeurons = numberOfInputNeurons + 1      
        self.numberOfHiddenNeurons = numberOfHiddenNeurons
        self.numberOfOutputNeurons = numberOfOutputNeurons        
        
        #Initialize the weights to 1.0
        self.weightsBetweenInputAndHiddenLayer = np.ones([self.numberOfInputNeurons,self.numberOfHiddenNeurons])
        self.weightsBetweenHiddenAndOutputLayer = np.ones([self.numberOfHiddenNeurons,self.numberOfOutputNeurons])
        
        #Initialize the activations to matrices of 1s
        self.activationsForInputLayer = np.ones([self.numberOfInputNeurons])
        self.activationsForHiddenLayer = np.ones([self.numberOfHiddenNeurons])
        self.activationsForOutputLayer = np.ones([self.numberOfOutputNeurons])
        
        #Vectorize the sigmoid
        self.vectorizedSigmoid = np.vectorize(self.sigmoid)
        self.vectorizedDifferentialOfSigmoid = np.vectorize(self.differentialOfSigmoid)
        
    def sigmoid(self,x):
#         return math.tanh(x)
        return 1/(1+(math.e**(-x)))
    
    def differentialOfSigmoid(self,y):
#         return 1 - y**2
        return (1-y) * y
    
    def computeOutput(self,inputVectors):                       
        #predictedOutput
        predictedOutputVector = 0
                
        for inputVector in inputVectors:
            self.activationsForInputLayer = inputVector[0,0]
            
            #add bias input
            self.activationsForInputLayer = np.insert(self.activationsForInputLayer,0,-1,axis=0)
            
            #hiddenLayer activations is sigmoid(dot product of inputActivation and inputWeights)
            self.activationsForHiddenLayer = self.vectorizedSigmoid(np.dot(self.activationsForInputLayer,self.weightsBetweenInputAndHiddenLayer))
            print "\nActivations for HiddenLayer"
            print "**************************"
            print self.activationsForHiddenLayer
            
            #outputLayer activations is sigmoid(dot product of hiddenActivation and hiddenWeights)
            self.activationsForOutputLayer = self.vectorizedSigmoid(np.dot(self.activationsForHiddenLayer,self.weightsBetweenHiddenAndOutputLayer))
            print "\nActivations for OutputLayer"
            print "**************************"
            print self.activationsForOutputLayer
            
            if(np.shape(predictedOutputVector) != ()):
                predictedOutputVector = np.append(predictedOutputVector,self.activationsForOutputLayer);                
            else:
                predictedOutputVector = self.activationsForOutputLayer
                
        print predictedOutputVector
        return predictedOutputVector                
    
    def train(self,trainingSet,iterations=1):
        print "**************************"
        print "TRAINING"
        print "**************************"     
        inputVectors = trainingSet[:,0]
        targetOutputVector = trainingSet[:,1]        
        for iteration in range(iterations):   
            predictedOutputVector = self.computeOutput(inputVectors)
            #TODO: Check error
            #TODO: Back propagate error
        
    def test(self,testingSet):
        print "**************************"
        print "TESTING"
        print "**************************"
        inputVectors = testingSet[:,0]
        self.patientLabels = testingSet[:,1]
        self.predictedPatientLabels = self.computeOutput(inputVectors)
        
        #Compute TP FP TN FN
#         self.computeTPTNFPFN()
        
        #Compute Accuracy
#         self.computeAccuracy()
        
        #Compute Sensitivity
        #Compute Specificity
        #Compute Precision
        #Compute Recall
        #Compute F-Measure
        #Compute MCC
        #Compute ROC
    
def main():
    neuralNet = NeuralNetwork(2,2,1)
    
    trainingSet = np.matrix([
                                 [[0,0],[0]],
                                 [[0,1],[0]],
                                 [[1,0],[0]],
                                 [[1,1],[1]]
                             ])
    
    neuralNet.train(trainingSet)
    neuralNet.test(trainingSet)
    
main();
    