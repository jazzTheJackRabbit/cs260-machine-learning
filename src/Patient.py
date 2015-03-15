import numpy as np
import math
class Patient:
	def __init__(self,filename):
		
		#measurements
		self.diastolicMeasurements = np.zeros([26,1])
		self.systolicMeasurements = np.zeros([26,1])
		
		#properties
		self.filename = ""
		self.classificationLabel = 0
		self.patientID = 0			
		self.predictedClassificationLabel = 0
		
		#features
		self.meanOfMeasurements = (0,0)
		self.varianceOfMeasurements = (0,0)
		self.standardDeviation = (0,0)	
		self.maxValue = (0,0)
		self.minValue = (0,0)
		self.differenceOfMaxAndMinValue = (0,0)
		self.skewnessOfMeasurements = (0,0)
		self.kurtosisOfMeasurements = (0,0)	
		self.pearsonsCorrelationCoefficient = (0,0)
		
		self.parseFileName(filename)
		
	def parseFileName(self,filename):
		self.filename = filename
		self.patientID = filename.split("/").pop().split(".")[0].split("_")[0]
		self.classificationLabel = filename.split("/").pop().split(".")[0].split("_")[1]
	
	def computeMean(self):
		meanForDiastolic = np.dot(np.transpose(self.diastolicMeasurements),np.ones([len(self.diastolicMeasurements),1]))[0,0]/len(self.diastolicMeasurements)
		meanForSystolic = np.dot(np.transpose(self.systolicMeasurements),np.ones([len(self.systolicMeasurements),1]))[0,0]/len(self.systolicMeasurements)
		self.meanOfMeasurements = (meanForDiastolic,meanForSystolic) 
		return self.meanOfMeasurements
	
	def computeVariance(self):
		matrixOfDifferencesForDiastolicMeasurements = np.subtract(self.diastolicMeasurements,self.meanOfMeasurements[0])
		matrixOfDifferencesForSystolicMeasurements = np.subtract(self.systolicMeasurements,self.meanOfMeasurements[1])
		varianceForDiastolicMeasurements = (np.dot(np.transpose(matrixOfDifferencesForDiastolicMeasurements),matrixOfDifferencesForDiastolicMeasurements)[0,0])/np.shape(self.diastolicMeasurements)[0]
		varianceForSystolicMeasurements = (np.dot(np.transpose(matrixOfDifferencesForSystolicMeasurements),matrixOfDifferencesForSystolicMeasurements)[0,0])/np.shape(self.systolicMeasurements)[0]		
		
		self.varianceOfMeasurements = (varianceForDiastolicMeasurements,varianceForSystolicMeasurements)				
		self.standardDeviation = (self.varianceOfMeasurements[0]**(0.5),self.varianceOfMeasurements[1]**(0.5))			
		
	def computeMaxMinDiff(self):
		self.maxValue = (np.amax(self.diastolicMeasurements),np.amax(self.systolicMeasurements))
		self.minValue = (np.amin(self.diastolicMeasurements),np.amin(self.systolicMeasurements))
		self.differenceOfMaxAndMinValue = (self.maxValue[0] - self.minValue[0],self.maxValue[1] - self.minValue[1])
	
	def computeSkewness(self):
		self.computeMean()
		self.computeVariance()
		
		differenceOfDiastolic = np.subtract(self.diastolicMeasurements,self.meanOfMeasurements[0])
		differenceOfSystolic = np.subtract(self.systolicMeasurements,self.meanOfMeasurements[1])
		
		skewnessOfDiastolic = ((np.dot(np.transpose(differenceOfDiastolic),np.multiply(differenceOfDiastolic,differenceOfDiastolic))/np.shape(differenceOfDiastolic)[0])/(self.standardDeviation[0]**3))[0,0]
		skewnessOfSystolic = ((np.dot(np.transpose(differenceOfSystolic),np.multiply(differenceOfSystolic,differenceOfSystolic))/np.shape(differenceOfSystolic)[0])/(self.standardDeviation[1]**3))[0,0]
		
		self.skewnessOfMeasurements = (skewnessOfDiastolic,skewnessOfSystolic)
		return self.skewnessOfMeasurements
	
	def computeKurtosis(self):
		self.computeMean()
		self.computeVariance()
		
		differenceOfDiastolic = np.subtract(self.diastolicMeasurements,self.meanOfMeasurements[0])
		differenceOfSystolic = np.subtract(self.systolicMeasurements,self.meanOfMeasurements[1])
		
		kurtosisOfDiastolic = ((np.dot(np.transpose(differenceOfDiastolic),np.multiply(differenceOfDiastolic,np.multiply(differenceOfDiastolic,differenceOfDiastolic)))/np.shape(differenceOfDiastolic)[0])/(self.standardDeviation[0]**4))[0,0]
		kurtosisOfSystolic = ((np.dot(np.transpose(differenceOfSystolic),np.multiply(differenceOfSystolic,np.multiply(differenceOfSystolic,differenceOfSystolic)))/np.shape(differenceOfSystolic)[0])/(self.standardDeviation[1]**4))[0,0]
		
		self.kurtosisOfMeasurements = (kurtosisOfDiastolic,kurtosisOfSystolic)
	
	def computePearsonsCorrelationCoefficient(self):
		self.computeMean()
		self.computeVariance()
		
		matrixOfDifferencesForDiastolicMeasurements = np.subtract(self.diastolicMeasurements,self.meanOfMeasurements[0])
		matrixOfDifferencesForSystolicMeasurements = np.subtract(self.systolicMeasurements,self.meanOfMeasurements[1])
		sumOfProductOfDifferences = (np.dot(np.transpose(matrixOfDifferencesForDiastolicMeasurements),matrixOfDifferencesForSystolicMeasurements)[0,0])
		pearsonsCorrelationCoefficient = (sumOfProductOfDifferences/(np.shape(self.diastolicMeasurements)[0]-1))		
		self.pearsonsCorrelationCoefficient = pearsonsCorrelationCoefficient
		return self.pearsonsCorrelationCoefficient
				

		
	
	def printMeasurements(self):
		print "\n##################\nDiastolic Measurements for Patient#"+str(self.patientID)+"\n##################\n"
		print self.diastolicMeasurements
		print "\n##################\nSystolic Measurements for Patient#"+str(self.patientID)+"\n##################\n"
		print self.systolicMeasurements
			