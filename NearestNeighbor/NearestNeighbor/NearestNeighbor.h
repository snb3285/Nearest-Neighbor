#ifndef NEARESTNEIGHBOR_H
#define NEARESTNEIGHBOR_H

#include <string>

float* loadData(std::string path);

void printSequence(int arr[], int size, std::stringstream& ss);

float kFoldCrossValidation(float* data, int* currentFeatureSet, int n);

void forwardSelection(float* data, std::string path);

void backwardElimination(float* data, std::string path);

#endif