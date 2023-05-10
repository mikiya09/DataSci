

import numpy as np 
import pandas as pd 
import random

# columns 
col = ['feature1', 'feature2', 'feature3', 'feature4',
       'feature5', 'feature6', 'feature7', 'feature8',
       'feature9', 'feature10', 'major']

df = pd.read_csv('./hw2data.csv', header=None, names=col)
print(df.head())


# ======================================= Question 1 ==================================================
print("Question 1:")
# function takes in dataPt of shape(M), suppVecs of shape(N, M+1), alphas of shape(N)
def svmClassify(dataPt, suppVecs, alphas, b):
    # sum up the difference of every first 10 columns in each row - new dataPt, and take the square of them 
    # and take the exp of final result 
    # because support vectors could have more than one
    kernelVals = np.exp(-np.sum((suppVecs[:, :10] - dataPt) ** 2, axis=1))
    result = b + np.sum(alphas * suppVecs[:, 10] * kernelVals)
    # choose whatever result that is greater than 1 as class 1
    if result >= 1:
        print(f"test result: {result}")
        return 1
    # otherwise class 0
    else:
        print(f"test result: {result}")
        return 0



# select a random row from the dataset
random_row = df.sample().to_numpy()[0]

# extract the features and label from the row
dataPt = random_row[:10]
label = random_row[10]

# select first three or all rows as support vectors
suppVecs = df.to_numpy()
# suppVecs = suppVecs[0:3]

# randomlly choose dimension of alpha based on the number of rows that this dataset has
alphas = np.random.rand(len(suppVecs))
# alphas = alphas[0:3]
print(f"alpha: {alphas}")

# randomlly choose any number as offset
b = np.random.rand()
print(f"offset: {b}")

# main function
predicted = svmClassify(dataPt, suppVecs, alphas, b)

# Print the true label and predicted class
print(f"True Class: {int(label)} | Predicted Class: {int(predicted)}")
print()



# ======================================= Question 2 ==================================================
def computeW(dataSet, labels, alphas):
    # multiply alphas and labels matrix
    alphas_times_labels = np.multiply(alphas, labels)
    # dot product of (alphas * labels) and dataSet
    w = np.dot(alphas_times_labels, dataSet)
    return w


# extract all lables from the dataset 
label = df.to_numpy()[:, -1]
# print(label)
# data point 
dataPt = df.to_numpy()[:, :-1]
# print(dataPt)

# weight 
w = computeW(dataPt, label, alphas)
print("Question 2:")
print(f"weight after computeW(): \n{w}")
print()



# ======================================= Question 3 ==================================================



def learnLam(dataTrain, iters):
    N, M = dataTrain.shape

    # step size 
    eps = 0.01
    # initialize lambda to 1's
    lambdas = np.ones(N)
    # initialize weight to 0's
    w = np.zeros(M - 1)
    # bias 
    b = 0
    
    # N iteration, which N = len(dataset) = data point = rows
    for _ in range(iters):
        for i in range(N):
            # feature vector of i-th data point
            x_i = dataTrain[i, :-1]
            # label of i-th data point
            y_i = dataTrain[i, -1]
            # dot product of W and x_i plus bias 
            wTx_i = np.dot(w, x_i) + b
            # make sure the calculation doesn't contain value less than 1, otherwise it will converage
            if y_i * wTx_i < 1:
                # update rule for lambda
                delta_lambda_i = eps * (1 - y_i * wTx_i)
                lambdas[i] += delta_lambda_i

        # compute the w for each i, updating it
        w = computeW(dataTrain[:, :-1], dataTrain[:, -1], lambdas)
        # update bias as well
        b = np.mean(dataTrain[:, -1] - np.dot(dataTrain[:, :-1], w))

    return lambdas
#



# dataTrain = np.random.rand(10, 4)

# setup dataTrain into numpy array
dataTrain = df.to_numpy()

# call the learnLam function with 5 iterations
lambdas = learnLam(dataTrain, 8)

# print the lambda values
print("Question 3:")
print(f'final lambdas: {lambdas}')



