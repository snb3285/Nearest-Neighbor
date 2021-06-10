#include <vector>
#include "NearestNeighbor.h"
using namespace std;

int main() {
	vector<vector<float>> data = loadData("Test/CS205_small_testdata__10.txt");
	forwardSelection(data);
	// backwardElimination(data);
	return 0;
}