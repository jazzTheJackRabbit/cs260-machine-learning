from Classifier import Classifier
import csv
import random
import math
import operator
import numpy as np

class NearestNeighbor(Classifier):
    def __init__(self,k):
        self.k = k
        
    def computeDistance(self,featureVector1,featureVector2):
        differenceMatrix = np.subtract(featureVector1,featureVector2)
        squaredDistance = np.dot(differenceMatrix,np.transpose(differenceMatrix))
        distance = np.sqrt(squaredDistance)
        return distance[0,0]
    
    #Find k nearest data points to the featureOfReference
    def findKNearestNeighbors(self,featureOfReference,trainingSet,k):
        #Returns k tuples in the format: (distance,featureVector)
        featureVectors = trainingSet[:,0]
        outputLabels = trainingSet[:,1]
        distancesAndFeatureTuples = []
        for featureIndex in range(len(featureVectors)):            
            featureVector = np.matrix(featureVectors[featureIndex][0,0])
            outputLabel = outputLabels[featureIndex,0]
            distance = self.computeDistance(featureOfReference, featureVector)
            distancesAndFeatureTuples.append((distance,featureVector,outputLabel))
        
        #sort the distancesAndFeatureTuples
        distancesAndFeatureTuples.sort(key=operator.itemgetter(0))#sort on the distance values
        
        #pick the top k distances as nearest neighbors        
        return distancesAndFeatureTuples[0:k]
             
    def computeClass(self,nearestNeighbors):
        classificationLabels = {}
        for nearestNeighborIndex in range(len(nearestNeighbors)):
            nearestNeighbor = nearestNeighbors[nearestNeighborIndex]            
            classLabelForNeighbor = nearestNeighbor[2][0]
            if(classificationLabels.has_key(classLabelForNeighbor)):
                classificationLabels[classLabelForNeighbor] += 1
            else:
                classificationLabels[classLabelForNeighbor] = 1
        #Sort the classificationLabels based on the count
        sorted_classificationLabels = sorted(classificationLabels.items(),key=operator.itemgetter(1),reverse=True) #sort on the counts
        #Pick the top class
        return sorted_classificationLabels[0][0]
                
    def classifyReferencePoint(self,featureOfReference,trainingSet,k):
        nearestNeighbors = self.findKNearestNeighbors(featureOfReference, trainingSet, k)
        classificationLabelForReferencePoint = self.computeClass(nearestNeighbors)
        return classificationLabelForReferencePoint
    
    def classify(self,trainingSet,testingSet,k=1):
        #TODO: change all k to self.k
        k=self.k            
        
        featureVectorsForTrainingSet = testingSet[:,0]
        actualOutputLabelsForTrainingSet = testingSet[:,1]
        
        #Setup the ACTUAL patient labels
        self.patientLabels = []
        for label in actualOutputLabelsForTrainingSet:
            self.patientLabels = np.append(self.patientLabels,label[0,0])
            
        predictedOutputVector = []              
        for referencePointIndex in range(len(featureVectorsForTrainingSet)):
            featureOfReference = np.matrix(featureVectorsForTrainingSet[referencePointIndex][0,0])            
            classLabelForReferencePoint = self.classifyReferencePoint(featureOfReference, trainingSet, k)
            predictedOutputVector = np.append(predictedOutputVector,classLabelForReferencePoint);                                
        self.predictedPatientLabels = predictedOutputVector
        
        self.printActualsVsPredictedLabels()
        self.evaluatePredictions()
            
        
def main():
    knn = NearestNeighbor(3)    

    #TODO: setup the Leave one out cross validation
#     trainingSet,testingSet = crossValidate(dataset)

    trainingSet = np.matrix([
                                [[1.5,1.5],[0]],
                                [[5,5],[1]],
                                [[3,3],[0]],
                                [[4,4],[1]]
                            ])
    testingSet = np.matrix([
                                [[1,1],[0]],
                                [[2,2],[0]],
                                [[6,6],[1]],                                
                            ])
        
    knn.classify(trainingSet,testingSet)
    
main();
     
        