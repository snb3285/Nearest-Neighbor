#ifndef NEARESTNEIGHBOR_H
#define NEARESTNEIGHBOR_H

#include <string>
#include <vector>

std::vector<std::vector<float>> loadData(std::string path);

void printSequence(std::vector<int>& vec);

float kFoldCrossValidation(std::vector<std::vector<float>> data, std::vector<int> currentFeatureSet);

void forwardSelection(std::vector<std::vector<float>> data);

void backwardElimination(std::vector<std::vector<float>> data);

#endif