
import numpy as np
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()

# Print the features and target variable
print("Features: ", iris.feature_names)
print("Target variable: ", iris.target_names)

# X = np.array(iris.data)
# y = np.array(iris.target)

# display data point and class
# print(X)
# print(y)


# implementation with sklearn library Naive Bayes
from sklearn.naive_bayes import GaussianNB

def classify_iris(sepal_length, sepal_width, petal_length, petal_width):
    # Load the IRIS dataset
    iris = load_iris()

    # Create a Gaussian Naive Bayes classifier
    clf = GaussianNB()

    # Train the classifier on the IRIS dataset
    clf.fit(iris.data, iris.target)

    # Create a feature vector for the new flower
    new_flower = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Use the trained classifier to predict the class of the new flower
    predicted_class = clf.predict(new_flower)

    # Convert the predicted class index to the corresponding class name
    class_names = iris.target_names.tolist()
    predicted_class_name = class_names[predicted_class[0]]

    return predicted_class_name

predicted_class = classify_iris(8.1, 2.5, 2.4, 0.2)
print(f'prediction with sklearn(class): {predicted_class}')



# implementation with only NUMPY 
def classify_iris_np(sepal_length, sepal_width, petal_length, petal_width):
    # Load the IRIS dataset
    iris = load_iris()

    # Split the dataset into features and labels
    X = iris.data
    y = iris.target

    # Compute the mean and variance of each feature for each class
    class_means = [np.mean(X[y == i], axis=0) for i in range(3)]
    class_variances = [np.var(X[y == i], axis=0) for i in range(3)]

    # Compute the class priors
    class_priors = np.bincount(y) / len(y)

    # Compute the likelihoods for each feature of the test example
    likelihoods = [[np.exp(-(x - mean)**2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)
                    for x, mean, variance in zip([sepal_length, sepal_width, petal_length, petal_width], class_means[i], class_variances[i])]
                   for i in range(3)]

    # Compute the joint probabilities for each class
    joint_probabilities = np.prod(likelihoods, axis=1) * class_priors

    # Normalize the joint probabilities to get the posterior probabilities
    posterior_probabilities = joint_probabilities / np.sum(joint_probabilities)

    # Return the class with the highest posterior probability
    predicted_class_index = np.argmax(posterior_probabilities)
    predicted_class_name = iris.target_names[predicted_class_index]

    return predicted_class_name

predicted_class_np = classify_iris_np(8.1, 2.5, 2.4, 0.2)
print(f'prediction with only np(class): {predicted_class_np}')
