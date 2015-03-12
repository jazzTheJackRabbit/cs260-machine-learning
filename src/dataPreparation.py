from Patient import Patient
import os
import numpy as np

def createPatientsInDirectory(directory):    
    allPatients = []    
    for dirname, dirnames, filenames in os.walk(directory):
        # print path to all filenames except .DS_Store.
        for filename in filenames:
            if(filename != ".DS_Store"):                
                patient = Patient(os.path.join(dirname, filename))
                allPatients.append(patient)
                
    return allPatients
    
def readPatientDataForFilesInDirectory(directory):    
    allPatients = createPatientsInDirectory(directory)
    
    for patientIndex in range(len(allPatients)):    
        filePath = allPatients[patientIndex].filename
        patientFileObject = open(filePath,'r')
        line = patientFileObject.readline()
        fileContents = []
        patientData = np.matrix([0,0])
        
        #Read all lines of the file.
        while(line):
            fileContents.append(line)
            line = patientFileObject.readline()
        
        #Remove title header.
        fileContents = fileContents[1:]
        for patientRowIndex in range(len(fileContents)):
            rowString = fileContents[patientRowIndex].split("\n")[0].split(",")
            row = np.matrix([int(rowString[1]),int(rowString[2])])
            if(patientRowIndex!=0):
                patientData = np.append(patientData,row,axis=0)
            else:
                patientData = row             
                                
        allPatients[patientIndex].diastolicMeasurements = patientData[:,[0]]
        allPatients[patientIndex].systolicMeasurements = patientData[:,[1]]
                                    
    return allPatients

def distanceForMeasurements(measurements):
    distance = 0
    for i in range(len(measurements)):
        distance = distance + measurements[i]**2
    
    distance = distance**(0.5)

#Main
allPatients = readPatientDataForFilesInDirectory('/Users/amogh/workspace/jazz/ucla/cs260a/MachineLearningProject/dataset/outDataClass')

print "\n##################\n Labels\n##################\n"
for patientIndex in range(len(allPatients)):     
    print allPatients[patientIndex].classificationLabel 

print "\n##################\n Compute Mean and Variance\n##################\n"
for patientIndex in range(len(allPatients)):    
    meanForPatientMeasurements = allPatients[patientIndex].computeMean()
    varianceOfMeasurements = allPatients[patientIndex].computeVariance()
    allPatients[patientIndex].computeMaxMinDiff()


print "\n##################\n Mean\n##################\n"
for patientIndex in range(len(allPatients)):
    print str((allPatients[patientIndex].meanOfMeasurements[0]))
    
print "\n##################\n Mean\n##################\n"
for patientIndex in range(len(allPatients)):
    print str((allPatients[patientIndex].meanOfMeasurements[1]))



print "\n##################\n Variance\n##################\n"
for patientIndex in range(len(allPatients)):
    print str((allPatients[patientIndex].varianceOfMeasurements[0]))
    
print "\n##################\n Variance\n##################\n"
for patientIndex in range(len(allPatients)):
    print str((allPatients[patientIndex].varianceOfMeasurements[1]))
    


print "\n##################\n SD/Mean\n##################\n"
for patientIndex in range(len(allPatients)):
    print str(((allPatients[patientIndex].meanOfMeasurements[0]))/(allPatients[patientIndex].varianceOfMeasurements[0]))
    
print "\n##################\n SD/Mean\n##################\n"
for patientIndex in range(len(allPatients)):
    print str(((allPatients[patientIndex].meanOfMeasurements[1]))/(allPatients[patientIndex].varianceOfMeasurements[1]))


print "\n##################\n SD/Mean \n##################\n"
for patientIndex in range(len(allPatients)):
    print "[["+str((allPatients[patientIndex].standardDeviation[0]/allPatients[patientIndex].meanOfMeasurements[0]))+","+str((allPatients[patientIndex].standardDeviation[1]/allPatients[patientIndex].meanOfMeasurements[1]))+"]"+",["+str(allPatients[patientIndex].classificationLabel)+"]],"



print "\n##################\n All Features\n##################\n"
for patientIndex in range(len(allPatients)):
    print "[["+str((allPatients[patientIndex].meanOfMeasurements[0]))+","+str((allPatients[patientIndex].meanOfMeasurements[1]))+","+str((allPatients[patientIndex].varianceOfMeasurements[0]))+","+str((allPatients[patientIndex].varianceOfMeasurements[1]))+","+str((allPatients[patientIndex].standardDeviation[0]/allPatients[patientIndex].meanOfMeasurements[0]))+","+str((allPatients[patientIndex].standardDeviation[1]/allPatients[patientIndex].meanOfMeasurements[1]))+"]"+",["+str(allPatients[patientIndex].classificationLabel)+"]],"

print "\n##################\n Deviation Between Max and Min\n##################\n"
for patientIndex in range(len(allPatients)):
    print "[["+str((allPatients[patientIndex].differenceOfMaxAndMinValue[0]))+","+str((allPatients[patientIndex].differenceOfMaxAndMinValue[1]))+"],["+str(allPatients[patientIndex].classificationLabel)+"]],"
    
print "\n##################\n Diastolic\n##################\n"
for patientIndex in range(len(allPatients)):
    print str((allPatients[patientIndex].differenceOfMaxAndMinValue[0]))

print "\n##################\n Systolic\n##################\n"
for patientIndex in range(len(allPatients)):
    print str((allPatients[patientIndex].differenceOfMaxAndMinValue[1]))    

print "\n##################\n Labels\n##################\n"
for patientIndex in range(len(allPatients)):   
    str(allPatients[patientIndex].classificationLabel)
        
# print "\n##################\n Mean\n##################\n"
# for patientIndex in range(len(allPatients)):
#     print "[["+str(int(allPatients[patientIndex].meanOfMeasurements[0]))+","+str(int(allPatients[patientIndex].meanOfMeasurements[1]))+"]"+",["+str(allPatients[patientIndex].classificationLabel)+"]],"
# 
# print "\n##################\n Variance\n##################\n"
# for patientIndex in range(len(allPatients)):
#     print "[["+str(int(allPatients[patientIndex].varianceOfMeasurements[0]))+","+str(int(allPatients[patientIndex].varianceOfMeasurements[1]))+"]"+",["+str(allPatients[patientIndex].classificationLabel)+"]],"
# 
# print "\n##################\n SD/Mean \n##################\n"
# for patientIndex in range(len(allPatients)):
#     print "[["+str((allPatients[patientIndex].standardDeviation[0]/allPatients[patientIndex].meanOfMeasurements[0]))+","+str((allPatients[patientIndex].standardDeviation[1]/allPatients[patientIndex].meanOfMeasurements[1]))+"]"+",["+str(allPatients[patientIndex].classificationLabel)+"]],"    
