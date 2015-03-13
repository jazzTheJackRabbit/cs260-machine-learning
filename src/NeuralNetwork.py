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
        
        #Initialize the weights to 1.0 and then randomize
        self.weightsBetweenInputAndHiddenLayer = np.zeros([self.numberOfInputNeurons,self.numberOfHiddenNeurons])
        self.weightsBetweenHiddenAndOutputLayer = np.zeros([self.numberOfHiddenNeurons,self.numberOfOutputNeurons])            
        
        #Initialize the activations to matrices of 1s
        self.activationsForInputLayer = np.ones([self.numberOfInputNeurons])
        self.activationsForHiddenLayer = np.ones([self.numberOfHiddenNeurons])
        self.activationsForOutputLayer = np.ones([self.numberOfOutputNeurons])
        
        #Vectorize the sigmoid
        self.vectorizedSigmoid = np.vectorize(self.sigmoid)
        self.vectorizedDifferentialOfSigmoid = np.vectorize(self.differentialOfSigmoid)
        
        #Previous update in weights for momentum   
        self.previousUpdateForInputHiddenLayers = np.zeros([self.numberOfInputNeurons,self.numberOfHiddenNeurons])
        self.previousUpdateForHiddenOutputLayers = np.zeros([self.numberOfHiddenNeurons,self.numberOfOutputNeurons]) 
        
        #Randomize Weights
        self.randomizeWeights(0.2, 2.0)
        
    def randomizeWeights(self,initialWeightIntensityBetweenInputAndHidden=0.2,initialWeightIntensityBetweenHiddenAndOutput=2):        
        for inputNeuronIndex in range(self.numberOfInputNeurons):
            for hiddenNeuronIndex in range(self.numberOfHiddenNeurons):
                self.weightsBetweenInputAndHiddenLayer[inputNeuronIndex,hiddenNeuronIndex] = ((initialWeightIntensityBetweenInputAndHidden - (-1*initialWeightIntensityBetweenInputAndHidden))*random.random()) + (-1*initialWeightIntensityBetweenInputAndHidden)
        for hiddenNeuronIndex in range(self.numberOfHiddenNeurons):
            for outputNeuronIndex in range(self.numberOfOutputNeurons):
                self.weightsBetweenHiddenAndOutputLayer[hiddenNeuronIndex,outputNeuronIndex] = ((initialWeightIntensityBetweenHiddenAndOutput - (-1*initialWeightIntensityBetweenHiddenAndOutput))*random.random()) + (-1*initialWeightIntensityBetweenHiddenAndOutput)
                
    def sigmoid(self,x):
#         return math.tanh(x)
        return 1/(1+(math.e**(-x)))
    
    def differentialOfSigmoid(self,y):
#         return 1 - y**2
        return (1-y) * y
    
    def computeOutput(self,inputVector):                                                               
        self.activationsForInputLayer = inputVector[0,0]
        
        #add bias input
        self.activationsForInputLayer = np.append(self.activationsForInputLayer,1.0)
        
        #hiddenLayer activations is sigmoid(dot product of inputActivation and inputWeights)
        self.activationsForHiddenLayer = self.vectorizedSigmoid(np.dot(self.activationsForInputLayer,self.weightsBetweenInputAndHiddenLayer))
#         print "\nActivations for HiddenLayer"
#         print "**************************"
#         print self.activationsForHiddenLayer
        
        #outputLayer activations is sigmoid(dot product of hiddenActivation and hiddenWeights)
        self.activationsForOutputLayer = self.vectorizedSigmoid(np.dot(self.activationsForHiddenLayer,self.weightsBetweenHiddenAndOutputLayer))
#         print "\nActivations for OutputLayer"
#         print "**************************"
#         print self.activationsForOutputLayer
        
    def backPropagation(self,targetOutputVectors, N_learningRate, M_momentum=0.3):
        
        #ERROR and DELTA computation : DELTA is GRADIENT * Error
        ########################################################
        outputDelta = np.zeros(np.shape(self.activationsForOutputLayer)[0])
        hiddenDelta = np.zeros(np.shape(self.activationsForHiddenLayer)[0])
        error = 0.0
        #for each outputNeuron
        for outputNeuronIndex in range(self.numberOfOutputNeurons):
            #Compute error between the target and predicted values
            error = targetOutputVectors[outputNeuronIndex,0] - self.activationsForOutputLayer[outputNeuronIndex]
            #Compute gradientOfThePredictedValue * error
            outputDelta[outputNeuronIndex] = error * self.vectorizedDifferentialOfSigmoid(self.activationsForOutputLayer[outputNeuronIndex])        
            
        #for each hidden neuron
        for hiddenNeuronIndex in range(self.numberOfHiddenNeurons):
            error = 0.0
            #Sum the errors from all the output neurons
            for outputNeuronIndex in range(self.numberOfOutputNeurons):
                error = error + self.weightsBetweenHiddenAndOutputLayer[hiddenNeuronIndex,outputNeuronIndex]*outputDelta[outputNeuronIndex]
            
            #Multiply the error with the gradientOfTheOutputAtTheHiddenLayer
            hiddenDelta[hiddenNeuronIndex] = error * self.vectorizedDifferentialOfSigmoid(self.activationsForHiddenLayer[hiddenNeuronIndex])            
                
        #WEIGHT UPDATES
        ###############
        
        #Update weights for hiddenLayer
        for hiddenNeuronIndex in range(self.numberOfHiddenNeurons):
            for outputNeuronIndex in range(self.numberOfOutputNeurons):                
                update = (outputDelta[outputNeuronIndex] * self.activationsForHiddenLayer[hiddenNeuronIndex])                
                self.weightsBetweenHiddenAndOutputLayer[hiddenNeuronIndex,outputNeuronIndex] = self.weightsBetweenHiddenAndOutputLayer[hiddenNeuronIndex,outputNeuronIndex] + (N_learningRate * update) + (M_momentum * self.previousUpdateForHiddenOutputLayers[hiddenNeuronIndex,outputNeuronIndex])
                self.previousUpdateForHiddenOutputLayers[hiddenNeuronIndex,outputNeuronIndex] = update
                      
        #Update weights for inputLayer
        for inputNeuronIndex in range(self.numberOfInputNeurons):
            for hiddenNeuronIndex in range(self.numberOfHiddenNeurons):
                #TODO: Check if the updates should be added or subtracted
                update = (hiddenDelta[hiddenNeuronIndex] * self.activationsForInputLayer[inputNeuronIndex])
                #TODO: Check if Momentum is required - to get out of local minima.
                self.weightsBetweenInputAndHiddenLayer[inputNeuronIndex,hiddenNeuronIndex] = self.weightsBetweenInputAndHiddenLayer[inputNeuronIndex,hiddenNeuronIndex]  + (N_learningRate * update) + (M_momentum * self.previousUpdateForInputHiddenLayers[inputNeuronIndex,hiddenNeuronIndex])
                self.previousUpdateForInputHiddenLayers[inputNeuronIndex,hiddenNeuronIndex] = update
                
        #Show the amount of error with squaredDifferenceError
        error = 0.0
        for outputNeuronIndex in range(self.numberOfOutputNeurons):
            error = error + ((0.5)*((targetOutputVectors[outputNeuronIndex,0] - self.activationsForOutputLayer[outputNeuronIndex])**2))        
        return error
    
    def train(self,trainingSet,iterations=1000,N_learningRate=0.1,M_momentum=0.3):
        #iterations are actually training epochs
        print "**************************"
        print "TRAINING"
        print "**************************"     
        inputVectors = trainingSet[:,0]
        targetOutputVector = trainingSet[:,1]                
        for iteration in range(iterations):    
            error = 0.0           
            predictedOutputVector = 0
            for inputVectorIndex in range(len(inputVectors)):
                inputVector = inputVectors[inputVectorIndex]
                self.computeOutput(inputVector)                
                if(np.shape(predictedOutputVector) != ()):
                    predictedOutputVector = np.append(predictedOutputVector,self.activationsForOutputLayer);                
                else:
                    predictedOutputVector = self.activationsForOutputLayer
                backPropChange = self.backPropagation(targetOutputVector[inputVectorIndex], N_learningRate,M_momentum)
#                 print "BackPropChange:"+str(backPropChange)
                error = error + backPropChange
            if iteration % 100 == 0:
                print 'error @iteration:'+ str(iteration) + ' =' + str(error)                            
#             print "Predicted Output(TRAINING):"
#             print predictedOutputVector
            
        
    def test(self,testingSet):
        print "**************************"
        print "TESTING"
        print "**************************"
        self.patientLabels = []
        inputVectors = testingSet[:,0]                        
        for label in testingSet[:,1]:
            self.patientLabels = np.append(self.patientLabels,label[0,0])
                    
        predictedOutputVector = 0
        for inputVector in inputVectors:
            self.computeOutput(inputVector)                
            if(np.shape(predictedOutputVector) != ()):
                predictedOutputVector = np.append(predictedOutputVector,self.activationsForOutputLayer);                
            else:
                predictedOutputVector = self.activationsForOutputLayer
        
        roundVector = np.vectorize(round)
        predictedOutputVector = roundVector(predictedOutputVector)
        self.predictedPatientLabels = predictedOutputVector            
                
        self.printActualsVsPredictedLabels()
        self.evaluatePredictions()
            
def main():
    neuralNet = NeuralNetwork(2,5,1)
    
    trainingSet = np.matrix([
                                [[0,0],[0]],
                                [[0,1],[1]],
                                [[1,0],[1]],
                                [[1,1],[0]]
                            ])
    
    testingSet = trainingSet
    
    neuralNet.train(trainingSet,iterations=1000,N_learningRate=0.3,M_momentum=0.4)    
    neuralNet.test(testingSet)
    
main();
    