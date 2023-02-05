
# 1) neighborClassifier 
#   a) Minues each of height values in the trainArray from the input value respectively
#   b) take the absolute value of each result
#   c) assign the label to the input value whose result is minimum
import numpy as np 

def neighborClassify(featureArray, trainArray):
    
    # classification label and labels(list) initialize
    label = 0
    labels = []
    # lopping through test data
    for test in featureArray:
        # set arithmetic difference to positive infinite
        result = float('inf')
        # looping through test data
        for train in trainArray:
            if abs(test - train[0]) < result:
                # update result
                result = abs(test - train[0])
                # select the minimum arithmetic difference pair
                label = train[1]

        # collect label
        labels.append(int(label))

    return labels

# classification: [6, 3, 9] -> [1, 0, 1]
featureArray = np.array([6, 3, 9])
# featureArray = np.array([6, 30, 9, 5.5, 7])
trainArray = np.array([[0.5, 0], [1.5, 0], [2.5, 0], [4.5, 1], [5, 1], [7.5, 0], [8, 1], [9.2, 1]])
print("NeighorChassify:")
print(neighborClassify(featureArray, trainArray))
print()
neighborClassify(featureArray, trainArray)


# 2) confusion
# [1, 1, 0, 0, 0, 1, 1, 1]  => trueLabels
# [0, 1, 1, 0, 0, 1, 0, 1]  => classifierOutput
# confusion matrix
#           lab=0 | lab=1
#           |-----|-----|
# true=0    |  2  |  1  |
# true=1    |  2  |  3  |
# ------------------------------
# check four conditions 
# [0, 0], [0, 1], [1, 0], [1, 1]

def confusion(classifierOutput, trueLabels):
    tn = 0 
    fn = 0
    fp = 0 
    tp = 0
    for i in range(len(trueLabels)):
        if trueLabels[i] == 0:
            if classifierOutput[i] == 0:
                tn += 1 
            else:
                fn += 1 
        elif trueLabels[i] == 1:
            if classifierOutput[i] == 0:
                fp += 1 
            else:
                tp += 1

    return np.array([[tn, fn], [fp, tp]])


# classifierOutput = np.array([0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
# trueLabels = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0])
classifierOutput = np.array([0, 1, 1, 0, 0, 1, 0, 1])
trueLabels = np.array([1, 1, 0, 0, 0, 1, 1, 1])
print("Confusion:")
print(confusion(classifierOutput, trueLabels))
print()
confusion(classifierOutput, trueLabels)



# 3) double Ones 
#   a) find the position where row with a 1 in the second column
#   b) copy the same item and add it after the that position
def doubleOnes(dataArray):
    count = 0 
    while True:
        if count == len(dataArray):
            break

        if dataArray[count][1] == 1:
            copy_item = np.copy(dataArray[count])
            dataArray = np.insert(dataArray, count, copy_item, axis=0) # add 2D array into 1D array, make sure on x-axis
            count += 1

        count += 1

    return dataArray

# [[5, 0], [3, 0], [8, 1], [10, 0], [4, 0], [2, 1]]
# [[5, 0], [3, 0], [8, 1], [8, 1], [10, 0], [4, 0], [2, 1]]
# [[5, 0], [3, 0], [8, 1], [8, 1], [10, 0], [4, 0], [2, 1], [2, 1]]

dataArray = np.array([[5, 0], [3, 0], [8, 1], [10, 0], [4, 0], [2, 1]])
# dataArray = np.array([[5, 0], [3, 0], [8, 1], [10, 0], [4, 0], [2, 1], [2, 2], [6, 1]])
expandedData = doubleOnes(dataArray)
print("doubleOnes:")
print(expandedData)
doubleOnes(dataArray)
