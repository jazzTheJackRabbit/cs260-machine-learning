from Classifier import Classifier
from DataPreparation import DataPreparation
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
        
        featureVectorsForTestingSet = testingSet[:,0]
        actualOutputLabelsForTestingSet = testingSet[:,1]
        
        #Setup the ACTUAL patient labels
#         self.patientLabels = []
#         for label in actualOutputLabelsForTestingSet:
#             self.patientLabels = np.append(self.patientLabels,label[0,0])
            
#         predictedOutputVector = []              
        for referencePointIndex in range(len(featureVectorsForTestingSet)):
            featureOfReference = np.matrix(featureVectorsForTestingSet[referencePointIndex][0,0])            
            classLabelForReferencePoint = self.classifyReferencePoint(featureOfReference, trainingSet, k)
#             predictedOutputVector = np.append(predictedOutputVector,classLabelForReferencePoint);                                
#         self.predictedPatientLabels = predictedOutputVector
#         
#         self.printActualsVsPredictedLabels()
#         self.evaluatePredictions()
            
        return classLabelForReferencePoint
        
def main():
           
    #Prepare Data 
    datasetToUse = "new"
    if(datasetToUse == "old"):
        dataPreparation = DataPreparation('/Users/amogh/workspace/jazz/ucla/cs260a/MachineLearningProject/dataset/outDataClass')
    else:
        dataPreparation = DataPreparation('/Users/amogh/workspace/jazz/ucla/cs260a/MachineLearningProject/dataset/outDataClassLDL')
    
    allPatients = dataPreparation.preparePatientData()
    
    #Uncomment based on whichever feature you want to select
#     datasetList = dataPreparation.computeFeatures("mean", allPatients)
#     datasetList = dataPreparation.computeFeatures("variance", allPatients)
#     datasetList = dataPreparation.computeFeatures("maxMinDiff", allPatients)
#     datasetList = dataPreparation.computeFeatures("skewness", allPatients)
#     datasetList = dataPreparation.computeFeatures("kurtosis", allPatients)    
    datasetList = dataPreparation.computeFeatures("pearsonsCorrelationCoefficient", allPatients)  
                  
    dataset = np.matrix(datasetList)
    
    for k in range(1,10):
        #Create the NearestNeighbor Model
        knn = NearestNeighbor(k)
            
        np.random.shuffle(dataset)
        predictedOutputVector = []     
        for i in range(len(dataset)):                
            #testingSet is top element
            referencePointLeftOutForTest = dataset[0]
            
            #pop top element
            dataset = np.delete(dataset,0,axis=0)
            
            #trainingSet is the rest
            trainingSet = dataset
            
            #create classification label list here.                
            classificationLabel = knn.classify(trainingSet,referencePointLeftOutForTest)
            predictedOutputVector = np.append(predictedOutputVector,classificationLabel);
            
            #add reference Point back into the dataset.
            dataset = np.append(dataset, referencePointLeftOutForTest,axis=0)
    
        knn.patientLabels = []
        for label in dataset[:,1]:
            knn.patientLabels = np.append(knn.patientLabels,label[0,0])
        #Flip the classifications        
        knn.predictedPatientLabels = predictedOutputVector
#         knn.predictedPatientLabels = (predictedOutputVector==0).astype(float)

            
        knn.evaluatePredictions()
        print "\nNumber of Neighbors = "+str(knn.k)    
        knn.printActualsVsPredictedLabels()    
    
main();
     
        