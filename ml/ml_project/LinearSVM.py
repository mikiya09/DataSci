
# basic Libraries
import pandas as pd 
import numpy as np
import time
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# LinearSVM class (gradient descent with regularization on hinge loss)
class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Get the number of samples and features from the input data
        n_samples, n_features = X.shape
        # Find the unique classes in the target variable
        self.classes_ = np.unique(y)
        # Initialize the classifiers list to store the weight and bias for each class
        self.classifiers_ = []

        for c in self.classes_:
            # Binary classification problem for linear SVM
            binary_y = np.where(y == c, 1, -1)
            # Initialize the weight vector with zeros
            w = np.zeros(n_features)
            # # Initialize bias to 0
            b = 0
            # Perform the gradient descent optimization
            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X):
                    # Check if the current sample satisfies the margin constraint
                    condition = binary_y[idx] * (np.dot(x_i, w) - b) >= 1
                    if condition:
                        # margin constraint satisified, update weight with only weight vector, no regularization
                        pass
                    else:
                        # If the margin constraint is not satisfied, update the weight vector use both the hinge loss
                        w -= self.lr * (- np.dot(x_i, binary_y[idx]))
                        # # Update the bias term using the hinge loss
                        b -= self.lr * binary_y[idx]

            self.classifiers_.append((w, b))


    
    # =============================== One Vs Rest ===================================
    # different regularization has different methods for programers' simplicity, actually just a line different
    # slack variable = w^2
    def fit_ovr_L(self, X, y):
        # Get the number of samples and features from the input data
        n_samples, n_features = X.shape
        # Find the unique classes in the target variable
        self.classes_ = np.unique(y)
        # Initialize the classifiers list to store the weight and bias for each class
        self.classifiers_ = []

        for c in self.classes_:
            # Binary classification problem for linear SVM
            binary_y = np.where(y == c, 1, -1)
            # Initialize the weight vector with zeros
            w = np.zeros(n_features)
            # # Initialize bias to 0
            b = 0
            # Perform the gradient descent optimization
            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X):
                    # Check if the current sample satisfies the margin constraint
                    condition = binary_y[idx] * (np.dot(x_i, w) - b) >= 1
                    if condition:
                        # margin constraint satisified, update weight with only weight vector (only L2 regularization not hinge loss)
                        w -= self.lr * (2 * self.lambda_param * w**2) # L2 regularization with slack variable w^2
                    else:
                        # If the margin constraint is not satisfied, update the weight vector use both the hinge loss and the L2 regularization
                        w -= self.lr * (2 * self.lambda_param * w**2 - np.dot(x_i, binary_y[idx]))
                        # # Update the bias term using the hinge loss
                        b -= self.lr * binary_y[idx]

            self.classifiers_.append((w, b))

    # L2 regurlarization
    def fit_ovr_L2(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.classifiers_ = []

        for c in self.classes_:
            # Binary classification problem for linear SVM
            binary_y = np.where(y == c, 1, -1)
            w = np.zeros(n_features)
            b = 0
            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X):
                    condition = binary_y[idx] * (np.dot(x_i, w) - b) >= 1
                    if condition:
                        w -= self.lr * (2 * self.lambda_param * w) # normal regularization with slack variable w
                    else:
                        w -= self.lr * (2 * self.lambda_param * w - np.dot(x_i, binary_y[idx]))
                        b -= self.lr * binary_y[idx]

            self.classifiers_.append((w, b))

    # L1 regularization 
    def fit_ovr_L1(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.classifiers_ = []

        for c in self.classes_:
            # Binary classification problem for linear SVM
            binary_y = np.where(y == c, 1, -1)
            w = np.zeros(n_features)
            b = 0
            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X):
                    condition = binary_y[idx] * (np.dot(x_i, w) - b) >= 1
                    if condition:
                        w -= self.lr * (self.lambda_param * np.sign(w)) # L1 regularization with slack variable sign(w)
                    else:
                        w -= self.lr * (2 * self.lambda_param * np.sign(w) - np.dot(x_i, binary_y[idx]))
                        b -= self.lr * binary_y[idx]

            self.classifiers_.append((w, b))

    # prediction on test set
    def predict_ovr(self, X):
        approximations = [np.dot(X, w) - b for w, b in self.classifiers_]
        approximations = np.array(approximations)
        return np.argmax(approximations, axis=0)


    # =============================== One Vs One =================================
    def fit_ovo_L2(self, X, y):
        # Get the number of samples and features from the input data
        n_samples, n_features = X.shape
        # Find the unique classes in the target label
        self.classes_ = np.unique(y)
        # unique classes
        n_classes = len(self.classes_)
        # Initialize the classifiers dictionary to store the weight and bias for each class pair
        self.classifiers_ = {}

        for i in range(n_classes):
            for j in range(i+1, n_classes):
                # Get the current pair of classes
                c1, c2 = self.classes_[i], self.classes_[j]

                # Filter samples for the current pair of classes
                idxs = np.where((y == c1) | (y == c2))
                X_filtered, y_filtered = X[idxs], y[idxs]
                # Create binary labels for the current pair of classes
                binary_y = np.where(y_filtered == c1, 1, -1)
                # Initialize the weight vector with zeros
                w = np.zeros(n_features)
                # Initialize bias to 0
                b = 0
                # Perform the gradient descent optimization
                for _ in range(self.n_iters):
                    for idx, x_i in enumerate(X_filtered):
                        # Check if the current sample satisfies the margin constraint
                        condition = binary_y[idx] * (np.dot(x_i, w) - b) >= 1
                        if condition:
                            # Update weight with only weight vector (only L2 regularization, not hinge loss)
                            w -= self.lr * (2 * self.lambda_param * w)
                        else:
                            # If the margin constraint is not satisfied, update the weight vector using both the hinge loss and the L2 regularization
                            w -= self.lr * (2 * self.lambda_param * w - np.dot(x_i, binary_y[idx]))
                            # Update the bias term using the hinge loss
                            b -= self.lr * binary_y[idx]

                # Store the weight and bias for the current pair of classes in the classifiers dictionary
                self.classifiers_[(c1, c2)] = (w, b)


    def predict_ovo(self, X):
        # Get the number of test samples
        n_samples = X.shape[0]
        # Initialize a matrix to store the votes for each class
        votes = np.zeros((n_samples, len(self.classes_)))

        # Iterate through the classifiers dictionary
        for (c1, c2), (w, b) in self.classifiers_.items():
            # Find the indices of the classes in the classes_ array
            class1_idx, class2_idx = np.where(self.classes_ == c1)[0][0], np.where(self.classes_ == c2)[0][0]
            # Calculate the decision function values for each test sample
            predictions = np.sign(np.dot(X, w) - b)
            # Increment the vote count for each class based on the sign of the decision function value
            votes[predictions == 1, class1_idx] += 1
            votes[predictions == -1, class2_idx] += 1

        # Return the class with the highest number of votes for each test sample
        return self.classes_[np.argmax(votes, axis=1)]


# loading dataset
data = pd.read_csv("./features_30_sec.csv")

# print out dataset 
# print(data.head())

# basic exploratory checking, everything is fine with our dataset
# print(f"• check missing value: \n{data.isnull().sum()}\n") # no missing value
# print(f"• check shape: {data.shape}\n") # check shape 
# print(f"• summarize data: {data.describe}") # summarization

# remove "filename" at the first column, the first column of the dataset is file name, doesn't remove from feature list 
data = data.iloc[0:, 1:] 

# exploratory analysis 
print(data['label'].value_counts())

# split dataset into X(features) and y(labels: genre)
y = data['label']
X = data.loc[:, data.columns != 'label']

# SVM is a classifier that used for processing numeric data, out dataset class labels are string value, so encoder them into numeric one
label_encoder = LabelEncoder() # create label encoder object 
label_encoder.fit(y) # fit classes 
y = label_encoder.transform(y) # transform string into numeric value [blues = 0, jazz = 1, etc]

# split the dataset into training and validating set (8:2 for training:testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Create MinMaxScaler to normalize feature into the range of (0, 1)
scaler = StandardScaler()
# Fit the scaler to the data and transform the data
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.fit_transform(X_test)

# check variance explained in terms of pca variance kept (self-defined function)
def pca_variance_explained(X, variance_threshold):
    # Center the data by subtracting the mean
    X_centered = X - X.mean()
    # Compute the Singular Value Decomposition (SVD)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Calculate the explained variance ratio
    explained_variance_ratio = (s ** 2) / np.sum(s ** 2)
    # Determine the number of components to retain based on the variance threshold
    cumsum_explained_variance = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumsum_explained_variance >= variance_threshold) + 1
    # Select the top n_components eigenvectors
    components = Vt[:n_components, :]
    # Project the data onto the reduced eigenvector space
    transformed_data = np.dot(X_centered, components.T)
    return transformed_data, n_components, components

# Fit PCA on the training data, I want to see how many principal components left if we keep certain amount of variance explained
X_train_pca, n_dimensions, components = pca_variance_explained(X_train_normalized, 0.95)
# Fit the test set as well 
X_test_pca = np.dot(X_test_normalized - X_test_normalized.mean(), components.T)
print("===========")
print("Principal Component Analysis with 95% variance kept, Number of dimensions retained: ", n_dimensions)



# catch the warning message, since using w^2 as slack variable create calculation overflow warning
# so I catch it to format the display warnings 
import warnings
# my warning message
overflow_warning_message = "*- Regularization with slack variable=w^2 create calculation overflow -*"
# function to replace the warning message
def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    print(f"{category.__name__}: {overflow_warning_message}")
# Replace the default warning handler with custom handler
warnings.showwarning = custom_warning_handler


print() # print separator
print("===========")
print("One Vs Rest Method:")

# Purely Hinge Loss 
svm = LinearSVM()
svm.fit(X_train_pca, y_train)
prediction = svm.predict_ovr(X_test_pca)
print(f"SVM baseline accuracy: {accuracy_score(y_test, prediction)}")


# Regularization w^2
svm0 = LinearSVM()
svm0.fit_ovr_L(X_train_pca, y_train)
prediction0 = svm0.predict_ovr(X_test_pca)
print(f"SVM Regularization with slack variable=w^2, accuracy: {accuracy_score(y_test, prediction)}")

# L1 regularization 
svm1 = LinearSVM()
svm1.fit_ovr_L1(X_train_pca, y_train)
prediction1 = svm1.predict_ovr(X_test_pca)
print("SVM L1 Regularization with slack variable=sign(w), accuracy: ", accuracy_score(y_test, prediction1))

# l2 regularization 
svm2 = LinearSVM()
svm2.fit_ovr_L2(X_train_pca, y_train)
prediction2 = svm2.predict_ovr(X_test_pca)
print("SVM L2 Regularization with slack variable=w, accuracy: ", accuracy_score(y_test, prediction2))

# L2 regularization without PCA
svm3 = LinearSVM()
svm3.fit_ovr_L2(X_train_normalized, y_train)
prediction3 = svm3.predict_ovr(X_test_normalized)
print("SVM L2 Regularization with slack variable=w without apply PCA, accuracy: ", accuracy_score(y_test, prediction3))

# One vs One method: L2 regularization
print("===========")
print("One Vs One Method:")
start_time = time.time()
svm4 = LinearSVM()
svm4.fit_ovo_L2(X_train_pca, y_train)
prediction4 = svm4.predict_ovo(X_test_pca)
end_time = time.time()
run_time = end_time - start_time 
print(f"SVM L2 Regularization with slack variable=w, accuracy: {accuracy_score(y_test, prediction4)}, runtime: {run_time}")