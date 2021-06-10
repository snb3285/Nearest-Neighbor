#include <iostream>
#include "NearestNeighbor.h"
using namespace std;

int main() {
	cout << "Enter the name of the file to use: ";
	string path;
	cin >> path;
	float* data = loadData(path);
	cout << "Enter the number for the algorithm you want to use:\n";
	cout << "(1) Forward Selection\n";
	cout << "(2) Backward Elimination\n";
	int choice;
	cin >> choice;
	if (choice == 1) {
		forwardSelection(data, path);
	}
	else if (choice == 2) {
		backwardElimination(data, path);
	}
	else {
		cout << "Invalid number\n";
	}
	delete[] data;
	return 0;
}