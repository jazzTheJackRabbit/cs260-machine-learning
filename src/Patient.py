import numpy as np

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
		varianceForDiastolicMeasurements = np.dot(np.transpose(matrixOfDifferencesForDiastolicMeasurements),matrixOfDifferencesForDiastolicMeasurements)[0,0]
		varianceForSystolicMeasurements = np.dot(np.transpose(matrixOfDifferencesForSystolicMeasurements),matrixOfDifferencesForSystolicMeasurements)[0,0]		
		
		self.varianceOfMeasurements = (varianceForDiastolicMeasurements,varianceForSystolicMeasurements)				
		self.standardDeviation = (self.varianceOfMeasurements[0]**(0.5),self.varianceOfMeasurements[1]**(0.5))			
		
	def computeMaxMinDiff(self):
		self.maxValue = (np.amax(self.diastolicMeasurements),np.amax(self.systolicMeasurements))
		self.minValue = (np.amin(self.diastolicMeasurements),np.amin(self.systolicMeasurements))
		self.differenceOfMaxAndMinValue = (self.maxValue[0] - self.minValue[0],self.maxValue[1] - self.minValue[1])
	
	def printMeasurements(self):
		print "\n##################\nDiastolic Measurements for Patient#"+str(self.patientID)+"\n##################\n"
		print self.diastolicMeasurements
		print "\n##################\nSystolic Measurements for Patient#"+str(self.patientID)+"\n##################\n"
		print self.systolicMeasurements