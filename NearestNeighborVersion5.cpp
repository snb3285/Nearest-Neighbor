#include <cmath>
#include <cfloat>
#include <climits>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <iomanip>
using namespace std;

constexpr int k = 5;

vector<vector<float>> loadData(string path) {
	string dataPoint;
	ifstream file(path);
	if (file.is_open()) {
		vector<vector<float>> data;
		while (getline(file, dataPoint)) {
			vector<float> splitData;
			stringstream ss(dataPoint);
			string temp;
			while (ss >> temp)
				splitData.push_back(stof(temp));
			data.push_back(splitData);
		}
		return data;
	}
	else {
		return vector<vector<float>>();
	}
}

void printSequence(vector<int>& vec) {
	if (vec.empty()) {
		cout << "{}";
		return;
	}
	cout << '{';
	for (auto it = vec.begin(); it != vec.end() - 1; it++)
		cout << *it << ", ";
	cout << *(vec.end() - 1) << '}';
}

float kFoldCrossValidation(vector<vector<float>> data, vector<int> currentFeatureSet) {
	float accuracyK = 0.0;
	int partitionSize = data.size() / k + 1;
	for (int i = 0; i < k; i++) {
		int numCorrect = 0;
		int start = i * partitionSize;
		int finish = min((i + 1) * partitionSize, (int)data.size());
		/* keep track of validation data indices for comparisons with test data
		add one to the integer division of the number of data points and k
		to get k partitions, with the last one being smaller if necessary */
		// iterate over validation data pointsand compare training data with validation data
		for (int valInd = start; valInd < finish; valInd++) {
			float nearestNeighborDistance = FLT_MAX;
			int nearestNeighborInd = INT_MAX;
			int nearestNeighborLabel = 0;
			int indLabel = data[valInd][0];
			for (int j = 0; j < data.size(); j++) {
				if (j < start || j >= finish) {
					float squareSum = 0.0;
					for (int m = 0; m < currentFeatureSet.size(); m++) {
						int feat = currentFeatureSet[m];
						float square = (data[valInd][feat] - data[j][feat]);
						square *= square;
						squareSum += square;
					}
					float distance = sqrt(squareSum);
					if (distance < nearestNeighborDistance) {
						nearestNeighborDistance = distance;
						nearestNeighborInd = j;
						nearestNeighborLabel = data[nearestNeighborInd][0];
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

void forwardSelection(vector<vector<float>> data) {
	/* keep track of best accuracyand feature for each level of the search tree
	there are as many levels in the search tree as number of features in the data
	start with an empty set and add the best performing feature at the end of each level */
	vector<int> currentFeatureSet;
	float acc = kFoldCrossValidation(data, currentFeatureSet);
	cout << "Current feature set: ";
	printSequence(currentFeatureSet);
	cout << '\n';
	cout << "Accuracy: " << setprecision(3) << acc * 100 << '%' << '\n';
	for (int i = 1; i < data[0].size(); i++) {
		float bestCurrentAcc = 0.0;
		int bestFeature = 0;
		cout << "Currently on search tree level " << i << '\n';
		for (int j = 1; j < data[0].size(); j++) {
			if (find(currentFeatureSet.begin(), currentFeatureSet.end(), j) == currentFeatureSet.end()) {
				cout << "Consider adding feature " << j << '\n';
				vector<int> temp(currentFeatureSet);
				temp.push_back(j);
				acc = kFoldCrossValidation(data, temp);
				cout << "Accuracy of feature set ";
				printSequence(temp);
				cout << " is " << acc * 100 << '%' << '\n';
				if (acc > bestCurrentAcc) {
					bestCurrentAcc = acc;
					bestFeature = j;
				}
			}
		}
		currentFeatureSet.push_back(bestFeature);
		cout << "Added feature " << bestFeature << " to the current feature set" << '\n';
		cout << "Current feature set: ";
		printSequence(currentFeatureSet);
		cout << '\n';
		cout << "Accuracy: " << setprecision(3) << bestCurrentAcc * 100 << '%' << '\n';
	}
}

void backwardElimination(vector<vector<float>> data) {
	/* keep track of best accuracy and feature for each level of the search tree
	there are as many levels in the search tree as number of features in the data
	start with all features in the set and remove the feature that results in best accuracy
	for the remaining set at the end of each level */
	vector<int> currentFeatureSet(data[0].size() - 1, 0);
	for (int i = 1; i < data[0].size(); i++)
		currentFeatureSet[i - 1] = i;
	float acc = kFoldCrossValidation(data, currentFeatureSet);
	cout << "Current feature set: ";
	printSequence(currentFeatureSet);
	cout << '\n';
	cout << "Accuracy: " << setprecision(3) << acc * 100 << '%' << '\n';
	for (int i = 1; i < data[0].size(); i++) {
		float bestCurrentAcc = 0.0;
		int bestFeatureToRemove = 0;
		cout << "Currently on search tree level " << i << '\n';
		for (int j = 0; j < currentFeatureSet.size(); j++) {
			cout << "Consider removing feature " << currentFeatureSet[j] << '\n';
			vector<int> temp(currentFeatureSet);
			temp.erase(find(temp.begin(), temp.end(), currentFeatureSet[j]));
			acc = kFoldCrossValidation(data, temp);
			cout << "Accuracy of feature set ";
			printSequence(temp);
			cout << " is " << acc * 100 << '%' << '\n';
			if (acc > bestCurrentAcc) {
				bestCurrentAcc = acc;
				bestFeatureToRemove = currentFeatureSet[j];
			}
		}
		vector<int>::iterator it = find(currentFeatureSet.begin(), currentFeatureSet.end(), bestFeatureToRemove);
		currentFeatureSet.erase(it);
		cout << "Removed feature " << bestFeatureToRemove << " from the current feature set" << '\n';
		cout << "Current feature set: ";
		printSequence(currentFeatureSet);
		cout << '\n';
		cout << "Accuracy: " << setprecision(3) << bestCurrentAcc * 100 << '%' << '\n';
	}
}