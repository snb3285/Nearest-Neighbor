import random
from math import sqrt

k = 4

def load_data(path):
    file = open(path, "r")
    data = [list(map(float, line.split())) for line in file]
    return data

def features_search(data):
    current_feature_set = []
    # keep track of best accuracy and feature for each level of the search tree
    # there are as many levels in the search tree as number of features in the data
    # len(data[0]) refers to the number of columns in the data
    print(len(data[0]))
    for i in range(1, len(data[0])):
        best_current_acc = 0
        best_feature = 0
        print("Currently on search tree level {}".format(i))
        for j in range(1, len(data[0])):
            if j not in current_feature_set:
                print("Consider adding feature {}".format(j))
                acc = k_fold_cross_validation(data, list(current_feature_set), j)
                if acc > best_current_acc:
                    best_current_acc = acc
                    best_feature = j
        current_feature_set.append(best_feature)
        print("Added feature {} to the current feature set".format(best_feature))
        print("Accuracy: {}".format(best_current_acc))
        print("Current feature set:", current_feature_set)

def k_fold_cross_validation(data, current_feature_set, feature_to_add):
    current_feature_set.append(feature_to_add)
    accuracy_k = []
    partition_size = len(data) // k + 1
    for i in range(k):
        num_correct = 0
        # keep track of validation data indices for comparisons with test data
        # add one to the integer division of the number of data points and k
        # to get k partitions, with the last one being smaller if necessary
        val_data_ind = list(range(i * partition_size, min((i + 1) * partition_size, len(data))))
        nearest_neighbor_distance = float("inf")
        nearest_neighbor_ind = float("inf")
        # iterate over validation data points and compare training data with validation data
        for ind in val_data_ind:
            ind_label = data[ind][0]
            for j in range(len(data)):
                if j not in val_data_ind:
                    squares = [(data[ind][feat] - data[j][feat]) ** 2 for feat in current_feature_set]
                    distance = sqrt(sum(squares))
                    if distance < nearest_neighbor_distance:
                        nearest_neighbor_distance = distance
                        nearest_neighbor_ind = j
                        nearest_neighbor_label = data[nearest_neighbor_ind][0]
            if ind_label == nearest_neighbor_label:
                num_correct += 1
        accuracy_k.append(num_correct / len(val_data_ind))
    accuracy = sum(accuracy_k) / len(accuracy_k)
    return accuracy
    
    
data = load_data("Test/CS205_small_testdata__10.txt")
features_search(data)
