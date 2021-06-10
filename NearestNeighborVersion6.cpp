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
				data[i * rows + j] = stof(temp);
			}
		}
		file.close();
		return data;
	}
	else {
		return new float[0];
	}
}

void printSequence(int arr[], int size, ofstream& output) {
	if (size == 0) {
		output << "{}";
		return;
	}
	output << '{';
	for (int i = 0; i < size - 1; i++)
		output << arr[i] << ", ";
	output << arr[size - 1] << '}';
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
			float nearestNeighborDistance = FLT_MAX;
			int nearestNeighborInd = INT_MAX;
			int nearestNeighborLabel = 0;
			int indLabel = data[valInd];
			for (int j = 0; j < cols; j++) {
				if (j < start || j >= finish) {
					float squareSum = 0.0;
					for (int m = 0; m < n; m++) {
						int feat = currentFeatureSet[m];
						float square = (data[feat * rows + valInd] - data[feat * rows + j]);
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
	string outputPath = filename + "_output.txt";
	ofstream output(outputPath);
	int* currentFeatureSet = new int[rows - 1];
	int* temp = new int[rows - 1];
	int n = 0;
	float acc = kFoldCrossValidation(data, currentFeatureSet, n);
	output << "Current feature set: ";
	printSequence(currentFeatureSet, n, output);
	output << '\n';
	output << "Accuracy: " << setprecision(3) << acc * 100 << '%' << '\n';
	for (int i = 1; i < rows; i++) {
		float bestCurrentAcc = 0.0;
		int bestFeature = 0;
		output << "Currently on search tree level " << i << '\n';
		for (int j = 1; j < rows; j++) {
			bool exists = false;
			for (int m = 0; m < n; m++) {
				if (currentFeatureSet[m] == j)
					exists = true;
			}
			if (!exists) {
				if (rows < featThreshold)
					output << "Consider adding feature " << j << '\n';
				temp[n] = j;
				acc = kFoldCrossValidation(data, temp, n + 1);
				if (rows < featThreshold) {
					output << "Accuracy of feature set ";
					printSequence(temp, n + 1, output);
					output << " is " << acc * 100 << '%' << '\n';
				}
				if (acc > bestCurrentAcc) {
					bestCurrentAcc = acc;
					bestFeature = j;
				}
			}
		}
		currentFeatureSet[n] = bestFeature;
		output << "Added feature " << bestFeature << " to the current feature set" << '\n';
		output << "Current feature set: ";
		n++;
		printSequence(currentFeatureSet, n, output);
		output << '\n';
		output << "Accuracy: " << setprecision(3) << bestCurrentAcc * 100 << '%' << "\n\n";
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
	string outputPath = filename + "_output.txt";
	ofstream output(outputPath);
	int* currentFeatureSet = new int[rows - 1];
	int* temp = new int[rows - 1];
	int n = rows - 1;
	for (int i = 1; i < rows; i++)
		currentFeatureSet[i - 1] = i;
	float acc = kFoldCrossValidation(data, currentFeatureSet, n);
	output << "Current feature set: ";
	printSequence(currentFeatureSet, n, output);
	output << '\n';
	output << "Accuracy: " << setprecision(3) << acc * 100 << '%' << '\n';
	for (int i = 1; i < rows; i++) {
		float bestCurrentAcc = 0.0;
		int bestFeatureToRemove = 0;
		output << "Currently on search tree level " << i << '\n';
		for (int j = 0; j < n; j++) {
			if (rows < featThreshold)
				output << "Consider removing feature " << currentFeatureSet[j] << '\n';
			bool skipped = false;
			for (int m = 0; m < n - 1; m++) {
				if (m == j)
					skipped = true;
				temp[m] = (skipped) ? currentFeatureSet[m + 1] : currentFeatureSet[m];
			}
			acc = kFoldCrossValidation(data, temp, n - 1);
			if (rows < featThreshold) {
				output << "Accuracy of feature set ";
				printSequence(temp, n - 1, output);
				output << " is " << acc * 100 << '%' << '\n';
			}
			if (acc > bestCurrentAcc) {
				bestCurrentAcc = acc;
				bestFeatureToRemove = currentFeatureSet[j];
			}
		}
		bool found = false;
		for (int j = 0; j < n; j++) {
			if (found)
				currentFeatureSet[j - 1] = currentFeatureSet[j];
			if (currentFeatureSet[j] == bestFeatureToRemove)
				found = true;
		}
		n--;
		currentFeatureSet[n] = 0;
		output << "Removed feature " << bestFeatureToRemove << " from the current feature set" << '\n';
		output << "Current feature set: ";
		printSequence(currentFeatureSet, n, output);
		output << '\n';
		output << "Accuracy: " << setprecision(3) << bestCurrentAcc * 100 << '%' << "\n\n";
	}
	output.close();
	delete[] currentFeatureSet;
	delete[] temp;
}