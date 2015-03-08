import os
import numpy as np

allFilePathList = []

for dirname, dirnames, filenames in os.walk('/Users/amogh/workspace/jazz/ucla/cs260a/MachineLearningProject/dataset/outDataClass'):
    # print path to all filenames.
    for filename in filenames:
        if(filename != ".DS_Store"):
            allFilePathList.append(os.path.join(dirname, filename))

allPatientData = []
for patientIndex in range(len(allFilePathList)):    
    filePath = allFilePathList[patientIndex]
    patientFileObject = open(filePath,'r')
    line = patientFileObject.readline()
    fileContents = []
    patientData = np.matrix([0,0])
    while(line):
        fileContents.append(line)
        line = patientFileObject.readline()
    fileContents = fileContents[1:]
    for patientRowIndex in range(len(fileContents)):
        rowString = fileContents[patientRowIndex].split("\n")[0].split(",")
        row = np.matrix([int(rowString[1]),int(rowString[2])])
        if(patientRowIndex!=0):
            patientData = np.append(patientData,row,axis=0)
        else:
            patientData = row             
            
    allPatientData.append(patientData)
                
print allPatientData

        