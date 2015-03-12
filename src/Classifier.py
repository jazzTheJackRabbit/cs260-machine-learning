import numpy as np
class Classifier:
	def __init__(self):
		self.accuracy = 0
		self.sensitivity = 0
		self.specificity = 0
		self.precision = 0
		self.recall = 0
		self.fmeasure = 0
		self.mcc = 0
		self.roc = 0
		
		self.truePositives = 0
		self.trueNegatives = 0
		self.falsePositives = 0
		self.falseNegatives = 0
		
		self.allPatients = []
		self.patientLabels = []
		self.predictedPatientLabels = []
		return 0
		
	def computeTPTNFPFN(self):
		self.truePositives = np.sum(np.dot(np.transpose(self.predictedPatientLabels),self.patientLabels))
		self.falseNegatives = np.sum(np.dot(np.transpose((self.predictedPatientLabels == 0).astype(int)),(self.patientLabels == 0).astype(int)))
		self.trueNegatives = np.sum(np.dot(np.transpose(self.predictedPatientLabels),self.patientLabels))
		self.truePositives = np.sum((np.transpose(self.predictedPatientLabels),(self.patientLabels == 0).astype(int)))
		
		print "TP:"+str(self.truePositives)
		print "TN:"+str(self.trueNegatives)
		print "FP:"+str(self.falsePositives)
		print "FN:"+str(self.falseNegatives)
	
	def computeAccuracy(self):
		self.accuracy = (self.truePositives + self.falsePositives)/(self.truePositives + self.trueNegatives + self.falseNegatives + self.falsePositives)
		print "Accuracy:"+str(self.accuracy)		
	
	def computeSensitivity(self):
		self.sensitivity = (self.truePositives)/(self.truePositives + self.falseNegatives)
	
	def computeSpecificity(self):
		self.sensitivity = (self.trueNegatives)/(self.trueNegatives + self.falsePositives)
	
	def computePrecision(self):
		self.precision = (self.truePositives)/(self.truePositives + self.falsePositives)
	
	def computeRecall(self):
		self.recall = (self.truePositives)/(self.truePositives + self.falseNegatives)
	
	def computeFMeasure(self):
		self.fmeasure = 2*(self.precision * self.recall)/(self.precision + self.recall)
	
	def MCC(self):
		self.mcc = ((self.truePositives*self.trueNegatives)-(self.falsePositives*self.falseNegatives))/(((self.truePositives + self.falsePositives)*(self.truePositives + self.falseNegatives)*(self.trueNegatives + self.falsePositives)*(self.trueNegatives + self.falseNegatives))**(0.5))
	
	def rocCharacteristics(self):
		return 0