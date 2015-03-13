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
		
		self.trainingSet = []
		self.testingSet = []
		return 0
		
	def computeTPTNFPFN(self):
		self.truePositives = np.sum(np.dot(np.transpose(self.predictedPatientLabels),self.patientLabels))
		self.trueNegatives = np.sum(np.dot(np.transpose((self.predictedPatientLabels == 0).astype(int)),(self.patientLabels == 0).astype(int)))
		self.falsePositives = np.sum(np.dot(np.transpose(self.predictedPatientLabels),(self.patientLabels == 0).astype(int)))
		self.falseNegatives = np.sum(np.dot(np.transpose((self.predictedPatientLabels == 0).astype(int)),self.patientLabels))
		
		print("")
		print "Positive Class:"
		print "TP:"+str(self.truePositives)		
		print "FP:"+str(self.falsePositives)
		print "Negative Class:"
		print "TN:"+str(self.trueNegatives)
		print "FN:"+str(self.falseNegatives)
		print("")
	
	def computeAccuracy(self):
		self.accuracy = (self.truePositives + self.trueNegatives)/(self.truePositives + self.trueNegatives + self.falseNegatives + self.falsePositives)
		print "Accuracy:"+str(self.accuracy*100)+"%"		
	
	def computeSensitivity(self):
		self.sensitivity = (self.truePositives)/(self.truePositives + self.falseNegatives)
		print "Sensitivity:"+str(self.sensitivity*100)+"%"
	
	def computeSpecificity(self):
		self.specificity = (self.trueNegatives)/(self.trueNegatives + self.falsePositives)
		print "Specificity:"+str(self.specificity*100)+"%"
	
	def computePrecision(self):
		self.precision = (self.truePositives)/(self.truePositives + self.falsePositives)
		print "Precision:"+str(self.precision*100)+"%"
	
	def computeRecall(self):
		self.recall = (self.truePositives)/(self.truePositives + self.falseNegatives)
		print "Recall:"+str(self.recall*100)+"%"
	
	def computeFMeasure(self):
		self.fmeasure = 2*(self.precision * self.recall)/(self.precision + self.recall)
		print "F-Measure:"+str(self.fmeasure*100)+"%"
	
	def MCC(self):
		self.mcc = ((self.truePositives*self.trueNegatives)-(self.falsePositives*self.falseNegatives))/(((self.truePositives + self.falsePositives)*(self.truePositives + self.falseNegatives)*(self.trueNegatives + self.falsePositives)*(self.trueNegatives + self.falseNegatives))**(0.5))
	
	def rocCharacteristics(self):
		return 0
	
	def evaluatePredictions(self):
		print "\n**********************"
		print "EVALUATING PREDICTIONS:"
		print "**********************"
		#Compute TP FP TN FN
		self.computeTPTNFPFN()

		#Compute Accuracy
		self.computeAccuracy()
		
		#Compute Sensitivity
		self.computeSensitivity()
		
		#Compute Specificity
		self.computeSpecificity()
		
		#Compute Precision
		self.computePrecision()
		
		#Compute Recall
		self.computeRecall()
		
		#Compute F-Measure
		self.computeFMeasure()
		
		#Compute MCC
		#Compute ROC
		
	def printActualsVsPredictedLabels(self):
		print "\nPredicted Output(TESTING):"
		print self.predictedPatientLabels
		
		print "Actual Output(TESTING):"
		print self.patientLabels		
		