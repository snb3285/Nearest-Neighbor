import random
from math import sqrt

def load_data(path):
    file = open(path, "r")
    data = [list(map(float, line.split())) for line in file]
    return data

def features_search(data):
    pass

def k_fold_cross_validation(data, current_feature_set, feature_to_add):
    pass
    
data = load_data("Test/CS205_small_testdata__10.txt")
features_search(data, [])
