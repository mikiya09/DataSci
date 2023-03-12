
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# =============================== Question 1 ==========================================


dataIn = pd.read_csv('./hw1Data.csv', names=['class', 'speed', 'chripDelay'])
#print(dataIn)
dataNP = dataIn.to_numpy()
# print(dataNP)

# check class 
# print(f"class label: {dataIn['class'].unique()}")

# seperate the speed value into different dataframe based on their own class 
class0 = dataIn[dataIn["class"] == 0]
class1 = dataIn[dataIn["class"] == 1]
class2 = dataIn[dataIn["class"] == 2]
class3 = dataIn[dataIn["class"] == 3]

# only choose speed feature for constructing histpgram
# but actually don't need to do in this way
# you can plot the graph directly by using class0['speed']
vector0 = class0['speed']
vector1 = class1['speed']
vector2 = class2['speed']
vector3 = class3['speed']

# vector ploting 
# 1) class 0 is close to exponential distribution
# 2) class 1 doesn't follows any of the three distribution
# 3) class 2 follows a Gaussian distribution 
# 4) class 3 follows a exponential decrease distribution
vectors = [vector0, vector1, vector2, vector3]

for i in vectors:
    plt.hist(i)
    plt.show()


# =============================== Question 2 ==========================================
testData = np.array([[0,0.5,200],[1,0.7,130],[0,0.2,100], [1,2,10], [0,1.5,50],[1,4,20]])
# print(testData.shape)


# Equation for lambda estimate  
# p(x|y) = mul(i)[lambda*exp(-lambda x^i)]
# ln((lambda|x_1, x_2, ..., x_n) = nln - lambda*sum(x_i))
# the final solution will be lambda = n / sum(x)

def learnParams(data):
    # collect all the unique class labels from the data
    # 1) assuming the the first columns are the attributes
    labels = np.unique(data[:, 0])

    # 2) assuming the label column has the attribute name 'class'
    # labels = data['class'].unique()
    numLabels = len(labels)

    # 3) features = all columns - class column
    features = data.shape[1] - 1

    # create an empty matrix, which has the dimension of classes as row, features as column
    params = np.zeros((numLabels, features))

    # Compute lambda for each class and feature
    for i in range(numLabels):
        # select data for current class
        current_feature = data[data[:, 0] == labels[i]][:, 1:]
        # print(f"data selected :\n{current_feature}")

        # calculate MLE estimate of lambda for speed
        n = len(current_feature)
        sum_speed = np.sum(current_feature[:, 0])
        # print(f"total speed value: {sum_speed}")
        lambda_speed = n / sum_speed
        # print(f"lambda speed: {lambda_speed}")

        # calculate MLE estimate of lambda for chirp delay
        sum_delay = np.sum(current_feature[:, 1])
        lambda_delay = n / sum_delay

        # store parameters for current class
        params[i, 0] = lambda_speed
        params[i, 1] = lambda_delay

    return params
# print(dataNP)
params = learnParams(dataNP)
print(f"trained lambda parameters: \n{params}\n")




# =============================== Question 3 ==========================================

testData2 = np.array([[0,0.5,200],[1,0.7,130],[0,0.2,100],[1,2,10],[0,1.5,50],[1,4,20]])


def learnPriors(data):
    numLabels = len(np.unique(data[:, 0]))
    # create an empty matrix for storing prior probability
    priors = np.zeros(numLabels)
    for i in range(numLabels):
        # sum the value belongs to the current class 
        # and divide it using the total number of the data
        priors[i] = np.sum(data[:, 0] == i) / len(data)
    return priors

priors = learnPriors(dataNP)
print(f"prior probability: \n{priors}\n")



# =============================== Question 4 ==========================================


def labelBayes(feature_array, lambdas, prior_probs):

    # compute the exponent for each class
    # create an empty holder for collecting labels
    exp = np.empty((feature_array.shape[0], lambdas.shape[0]))
    # get dimensions
    for i in range(lambdas.shape[0]):
        # calculating the 
        exp[:, i] = np.sum(-lambdas[i] * feature_array + np.log(lambdas[i]), axis=1) + np.log(prior_probs[i])

    # Determine the predicted class label for each feature
    pred_labels = np.argmax(exp, axis=1)

    return pred_labels





# features = np.array([[3.4, 20], [0.72, 60], [0.74, 2.04], [2.56, 8.5]])
features = np.array([[3.4, 20], [0.72, 60], [0.74, 2.04], [3.4549, 119.802]])
# params = np.array([[0.7, 0.2], [0.4, 0.1]])
# priors = np.array([0.4, 0.6])

print(f"classification: \n{labelBayes(features, params, priors)}")
