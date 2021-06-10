from math import sqrt

k = 5

def load_data(path):
    file = open(path, "r")
    data = [list(map(float, line.split())) for line in file]
    return data

def forward_selection(data):
    current_feature_set = set()
    # keep track of best accuracy and feature for each level of the search tree
    # there are as many levels in the search tree as number of features in the data
    # len(data[0]) refers to the number of columns in the data
    #print(len(data[0]))
    acc = k_fold_cross_validation(data, current_feature_set)
    print("Current feature set:", current_feature_set)
    print("Accuracy:", acc)
    for i in range(1, len(data[0])):
        best_current_acc = 0
        best_feature = 0
        print("Currently on search tree level {}".format(i))
        for j in range(1, len(data[0])):
            if j not in current_feature_set:
                print("Consider adding feature {}".format(j))
                temp = set(current_feature_set)
                temp.add(j)
                acc = k_fold_cross_validation(data, temp)
                if acc > best_current_acc:
                    best_current_acc = acc
                    best_feature = j
        current_feature_set.add(best_feature)
        print("Added feature {} to the current feature set".format(best_feature))
        print("Current feature set:", current_feature_set)
        print("Accuracy:", best_current_acc)

def backward_elimination(data):
    current_feature_set = set(range(1, len(data[0])))
    acc = k_fold_cross_validation(data, current_feature_set)
    print("Current feature set:", current_feature_set)
    print("Accuracy:", acc)
    for i in range(1, len(data[0])):
        best_current_acc = 0
        best_feature_remove = 0
        print("Currently on search tree level {}".format(i))
        for feat in current_feature_set:
            print("Consider removing feature {}".format(feat))
            temp = set(current_feature_set)
            temp.remove(feat)
            acc = k_fold_cross_validation(data, temp)
            if acc > best_current_acc:
                best_current_acc = acc
                best_feature_remove = feat
        current_feature_set.remove(best_feature_remove)
        print("Removed feature {} from the current feature set".format(best_feature_remove))
        print("Current feature set:", current_feature_set)
        print("Accuracy:", best_current_acc)
            
    

def k_fold_cross_validation(data, current_feature_set):
    accuracy_k = []
    partition_size = len(data) // k + 1
    for i in range(k):
        num_correct = 0
        # keep track of validation data indices for comparisons with test data
        # add one to the integer division of the number of data points and k
        # to get k partitions, with the last one being smaller if necessary
        val_data_ind = list(range(i * partition_size, min((i + 1) * partition_size, len(data))))
        # iterate over validation data points and compare training data with validation data
        for ind in val_data_ind:
            nearest_neighbor_distance = float("inf")
            nearest_neighbor_ind = float("inf")
            nearest_neighbor_label = 0
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
    
    
data = load_data("Real/CS205_large_testdata__32.txt")
forward_selection(data)
#backward_elimination(data)
