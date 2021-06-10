#include <cmath>
#include <cfloat>
#include <climits>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
using namespace std;

constexpr int k = 5;
constexpr int featThreshold = 20;

int rows = 0;
int cols = 0;

float* loadData(string path) {
	string dataPoint;
	ifstream file(path);
	if (file.is_open()) {
		getline(file, dataPoint);
		stringstream ss(dataPoint);
		string temp;
		rows = 0;
		while (ss >> temp)
			rows++;
		cols = 1;
		while (getline(file, dataPoint)) {
			cols++;
		}
		float* data = new float[rows * cols];
		file.clear();
		file.seekg(0, file.beg);
		for (int j = 0; j < cols; j++) {
			getline(file, dataPoint);
			stringstream ss(dataPoint);
			string temp;
			for (int i = 0; i < rows; i++) {
				ss >> temp;
				data[i * cols + j] = stod(temp);
			}
		}
		file.close();
		return data;
	}
	else {
		return new float[0];
	}
}

void printSequence(int arr[], int size, stringstream& ss) {
	if (size == 0) {
		ss << "{}";
		return;
	}
	ss << '{';
	for (int i = 0; i < size - 1; i++)
		ss << arr[i] << ", ";
	ss << arr[size - 1] << '}';
}

// n is the size of currentFeatureSet
float kFoldCrossValidation(float* data, int* currentFeatureSet, int n) {
	float accuracyK = 0.0;
	int partitionSize = cols / k + 1;
	for (int i = 0; i < k; i++) {
		int numCorrect = 0;
		int start = i * partitionSize;
		int finish = min((i + 1) * partitionSize, cols);
		/* keep track of validation data indices for comparisons with test data
		add one to the integer division of the number of data points and k
		to get k partitions, with the last one being smaller if necessary */
		// iterate over validation data pointsand compare training data with validation data
		for (int valInd = start; valInd < finish; valInd++) {
			float nearestNeighborDistance = DBL_MAX;
			int nearestNeighborInd = INT_MAX;
			int nearestNeighborLabel = 0;
			int indLabel = data[valInd];
			for (int j = 0; j < cols; j++) {
				if (j < start || j >= finish) {
					float squareSum = 0.0;
					for (int m = 0; m < n; m++) {
						int feat = currentFeatureSet[m];
						float square = (data[feat * cols + valInd] - data[feat * cols + j]);
						square *= square;
						squareSum += square;
					}
					float distance = sqrt(squareSum);
					if (distance < nearestNeighborDistance) {
						nearestNeighborDistance = distance;
						nearestNeighborInd = j;
						nearestNeighborLabel = data[nearestNeighborInd];
					}
				}
			}
			if (indLabel == nearestNeighborLabel) {
				numCorrect++;
			}
		}
		accuracyK += (float)numCorrect / (finish - start);
	}
	float accuracy = accuracyK / k;
	return accuracy;
}

void forwardSelection(float* data, string path) {
	/* keep track of best accuracyand feature for each level of the search tree
	there are as many levels in the search tree as number of features in the data
	start with an empty set and add the best performing feature at the end of each level */
	// n is the size of currentFeatureSet
	stringstream ss(path);
	string filename;
	getline(ss, filename, '.');
	string outputPath = filename + "_output_forward.txt";
	ofstream output(outputPath);
	ss.str("");
	int* currentFeatureSet = new int[rows - 1];
	int* temp = new int[rows - 1];
	int n = 0;
	float acc = kFoldCrossValidation(data, currentFeatureSet, n);
	ss << "Current feature set: ";
	printSequence(currentFeatureSet, n, ss);
	ss << '\n';
	ss << "Accuracy: " << setprecision(3) << acc * 100 << '%' << "\n\n";
	for (int i = 1; i < rows; i++) {
		float bestCurrentAcc = 0.0;
		int bestFeature = 0;
		ss << "Currently on search tree level " << i << '\n';
		for (int j = 1; j < rows; j++) {
			bool exists = false;
			for (int m = 0; m < n; m++) {
				if (currentFeatureSet[m] == j)
					exists = true;
			}
			if (!exists) {
				if (rows < featThreshold)
					ss << "Consider adding feature " << j << '\n';
				temp[n] = j;
				acc = kFoldCrossValidation(data, temp, n + 1);
				if (rows < featThreshold) {
					ss << "Accuracy of feature set ";
					printSequence(temp, n + 1, ss);
					ss << " is " << acc * 100 << '%' << '\n';
				}
				if (acc > bestCurrentAcc) {
					bestCurrentAcc = acc;
					bestFeature = j;
				}
			}
		}
		currentFeatureSet[n] = bestFeature;
		temp[n] = bestFeature;
		n++;
		ss << "Added feature " << bestFeature << " to the current feature set" << '\n';
		ss << "Current feature set: ";
		printSequence(currentFeatureSet, n, ss);
		ss << '\n';
		ss << "Accuracy: " << setprecision(3) << bestCurrentAcc * 100 << '%' << "\n\n";
		output << ss.rdbuf();
	}
	output.close();
	delete[] currentFeatureSet;
	delete[] temp;
}

void backwardElimination(float* data, string path) {
	/* keep track of best accuracy and feature for each level of the search tree
	there are as many levels in the search tree as number of features in the data
	start with all features in the set and remove the feature that results in best accuracy
	for the remaining set at the end of each level */
	// n is the size of currentFeatureSet
	stringstream ss(path);
	string filename;
	getline(ss, filename, '.');
	string outputPath = filename + "_output_backward.txt";
	ofstream output(outputPath);
	ss.str("");
	int* currentFeatureSet = new int[rows - 1];
	int* temp = new int[rows - 1];
	int n = rows - 1;
	for (int i = 1; i < rows; i++) {
		currentFeatureSet[i - 1] = i;
		temp[i - 1] = i;
	}
	float acc = kFoldCrossValidation(data, currentFeatureSet, n);
	ss << "Current feature set: ";
	printSequence(currentFeatureSet, n, ss);
	ss << '\n';
	ss << "Accuracy: " << setprecision(3) << acc * 100 << '%' << "\n\n";
	for (int i = 1; i < rows; i++) {
		float bestCurrentAcc = 0.0;
		int bestFeatureToRemove = 0;
		ss << "Currently on search tree level " << i << '\n';
		for (int j = 0; j < n; j++) {
			if (rows < featThreshold)
				ss << "Consider removing feature " << currentFeatureSet[j] << '\n';
			bool skipped = false;
			for (int m = 0; m < n - 1; m++) {
				if (m == j)
					skipped = true;
				temp[m] = (skipped) ? currentFeatureSet[m + 1] : currentFeatureSet[m];
			}
			acc = kFoldCrossValidation(data, temp, n - 1);
			if (rows < featThreshold) {
				ss << "Accuracy of feature set ";
				printSequence(temp, n - 1, ss);
				ss << " is " << acc * 100 << '%' << '\n';
			}
			if (acc > bestCurrentAcc) {
				bestCurrentAcc = acc;
				bestFeatureToRemove = currentFeatureSet[j];
			}
		}
		bool found = false;
		for (int j = 0; j < n; j++) {
			if (found) {
				currentFeatureSet[j - 1] = currentFeatureSet[j];
			}
			if (currentFeatureSet[j] == bestFeatureToRemove)
				found = true;
		}
		n--;
		for (int j = 0; j < n; j++)
			temp[j] = currentFeatureSet[j];
		currentFeatureSet[n] = 0;
		temp[n] = 0;
		ss << "Removed feature " << bestFeatureToRemove << " from the current feature set" << '\n';
		ss << "Current feature set: ";
		printSequence(currentFeatureSet, n, ss);
		ss << '\n';
		ss << "Accuracy: " << setprecision(3) << bestCurrentAcc * 100 << '%' << "\n\n";
		output << ss.rdbuf();
	}
	output.close();
	delete[] currentFeatureSet;
	delete[] temp;
}