Instructions to run the algorithms.

The Directory structure for the src/ folder is as follows:

    src/
    ├── Classifier.py
    ├── DataPreparation.py
    ├── NearestNeighbor.py
    ├── NeuralNetwork.py
    └── Patient.py

To run the neural network:
- Run the following command:
	python NeuralNetwork.py

To run the nearest neighbor model
-Run the following command:
	python NearestNeighbor.py


The main() functions in both the files will take care of reading the datasets. The code runs as follows for:
	
Neural Net:

	for each dataset, including old and new:
		for each featue set in (Mean, Variance, MaxMinDiff, Skewness, Kurtosis, Pearson's Correlation Coefficient):
			for each neural network configuration:
				create a new neural net
				for each of the cross validation steps
					classify

K-NN: 

	You will only need to change the 'datasetToUse = "old"' variable in the main function to "new" if the new dataset is to be used.
	The code is written to run for increasing values of k in the range of 1 to 10.


If you have any questions, please email me.