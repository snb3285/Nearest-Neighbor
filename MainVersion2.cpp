#include <iostream>
#include "NearestNeighbor.h"
using namespace std;

int main() {
	cout << "Enter the name of the file to use: ";
	string path;
	cin >> path;
	float* data = loadData(path);
	forwardSelection(data, path);
	// backwardElimination(data, path);
	delete[] data;
	return 0;
}