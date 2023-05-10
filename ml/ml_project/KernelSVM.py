# basic Libraries
import pandas as pd 
import numpy as np
import time
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class KernelSVM:
    def __init__(self, C=1, kernel_type="linear", gamma=None):
        self.C = C
        self.kernel_type = kernel_type
        self.kernel_func = self.get_kernel_func(kernel_type)
        self.gamma = gamma
        self.classifiers = None

    def get_kernel_func(self, kernel_type):
        if kernel_type == "linear":
            return linear_kernel
        elif kernel_type == "rbf":
            return rbf_kernel
        elif kernel_type == "polynomial":
            return polynomial_kernel
        elif kernel_type == "laplace_rbf":
            return laplace_rbf_kernel
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

    def train_svm(self, X, y, C, tol=1e-3, max_passes=5):
        # Initialize alphas and b
        alphas = np.zeros(X.shape[0])
        b = 0.0

        # Set up the Gram matrix
        # K = np.array([[self.kernel_func(x1, x2, self.gamma) for x2 in X] for x1 in X])
        if self.kernel_type == "linear":
            K = np.array([[self.kernel_func(x1, x2) for x2 in X] for x1 in X])
        else:
            K = np.array([[self.kernel_func(x1, x2, self.gamma) for x2 in X] for x1 in X])


        passes = 0
        while passes < max_passes:
            num_changed_alphas = 0
            for i in range(X.shape[0]):
                # Calculate the error Ei for the current example i
                Ei = b + np.sum(alphas * y * K[:, i]) - y[i]

                # Check if the error Ei violates the KKT conditions
                if (y[i] * Ei < -tol and alphas[i] < C) or (y[i] * Ei > tol and alphas[i] > 0):
                    # Randomly select an index j not equal to i
                    j = np.random.choice([k for k in range(X.shape[0]) if k != i])

                    # Calculate the error Ej for the selected example j
                    Ej = b + np.sum(alphas * y * K[:, j]) - y[j]

                    # Store the old alphas for i and j
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]

                    # Calculate the lower and upper bounds L and H for alpha_j
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - C)
                        H = min(C, alphas[i] + alphas[j])

                    # If L and H are equal, skip the current iteration
                    if L == H:
                        continue

                    # Calculate eta, the second derivative of the objective function
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]

                    # If eta is non-negative, skip the current iteration
                    if eta >= 0:
                        continue

                    # Update alpha_j using the formula
                    alphas[j] -= y[j] * (Ei - Ej) / eta
                    alphas[j] = np.clip(alphas[j], L, H)

                    # If the change in alpha_j is too small, skip the current iteration
                    if abs(alphas[j] - alpha_j_old) < tol:
                        continue

                    # Update alpha_i using the formula
                    alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                    # Calculate the new bias term b1 and b2
                    b1 = b - Ei - y[i] * (alphas[i] - alpha_i_old) * K[i, i] - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                    b2 = b - Ej - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                    # Set b to the appropriate value based on the updated alphas
                    if 0 < alphas[i] < C:
                        b = b1
                    elif 0 < alphas[j] < C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    num_changed_alphas += 1

            # If no alphas were updated, increment passes, else reset it
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        # Identify the support vectors (data points for which alpha > 0)
        support_vector_indices = alphas > tol
        support_vectors = X[support_vector_indices]
        support_alphas = alphas[support_vector_indices]
        support_labels = y[support_vector_indices]

        return support_vectors, support_alphas, support_labels, b


    def svm_predict(self, X, support_vectors, support_alphas, support_labels, b, kernel_func, gamma):
        # Compute the kernel matrix between the test points and the support vectors
        # K = np.array([[self.kernel_func(x1, x2, self.gamma) for x2 in support_vectors] for x1 in X])
        if self.kernel_type == "linear":
            K = np.array([[kernel_func(x1, x2) for x2 in support_vectors] for x1 in X])
        else:
            K = np.array([[kernel_func(x1, x2, gamma) for x2 in support_vectors] for x1 in X])

        # Compute the predicted labels using the support vectors, alphas, b, and the kernel matrix
        y_pred = np.dot(K, support_alphas * support_labels) + b

        return np.sign(y_pred)


    def train_all_svms(self, X, y, C):
        classifiers = {}
        unique_labels = np.unique(y)
        
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i + 1:]:
                # Create a binary label encoding for the current pair of classes
                binary_y = y[np.isin(y, [label1, label2])]
                binary_X = X[np.isin(y, [label1, label2])]
                binary_y_encoded = np.where(binary_y == label1, -1, 1)

                # Train the SVM for the current pair of classes
                support_vectors, support_alphas, support_labels, b = self.train_svm(binary_X, binary_y_encoded, C)
                classifiers[(label1, label2)] = (support_vectors, support_alphas, support_labels, b)

        return classifiers

    # named with ovr, but actually is one-vs-one method
    def svm_predict_ovr(self, X, classifiers):
        unique_labels = sorted(list(set([label for pair in classifiers.keys() for label in pair])))
        votes = np.zeros((X.shape[0], len(unique_labels)))

        for (label1, label2), (support_vectors, support_alphas, support_labels, b) in classifiers.items():
            # Predict the class for each pair of classes and count the votes for each class
            # y_pred_encoded = self.svm_predict(X, support_vectors, support_alphas, support_labels, b)
            y_pred_encoded = self.svm_predict(X, support_vectors, support_alphas, support_labels, b, self.kernel_func, self.gamma)
            y_pred = np.where(y_pred_encoded == -1, label1, label2)
            for i, label in enumerate(unique_labels):
                votes[:, i] += (y_pred == label)

        # Return the class with the most votes for each test point
        return np.array(unique_labels)[np.argmax(votes, axis=1)]


    def fit(self, X, y):
        self.classifiers = self.train_all_svms(X, y, self.C)

    def predict(self, X):
        return self.svm_predict_ovr(X, self.classifiers)
    
    def cross_val_score(self, X, y, C, k=5):
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = []

        for train_indices, val_indices in kf.split(X):
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            self.C = C  # Update the instance variable C
            self.fit(X_train, y_train)  # Call the fit method to train the model
            y_pred = self.predict(X_val)
            accuracy = np.mean(y_val == y_pred)
            scores.append(accuracy)

        return np.mean(scores)



def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def rbf_kernel(x1, x2, gamma):
    # print("Gamma: ", gamma)
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

# here you are actually passing degree not gamma to the function, but for convinient, I didn't change the parameter name
# in the main function, so just remember that in the following parameter where it says gamma=2, it acutally mean degree=2
def polynomial_kernel(x1, x2, degree):
    return (np.dot(x1, x2) + 1) ** degree

def laplace_rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.sum(np.abs(x1 - x2)))


# loading dataset
data = pd.read_csv("./features_30_sec.csv")

# remove "filename" at the first column, the first column of the dataset is file name, doesn't remove from feature list 
data = data.iloc[0:, 1:] 

# split dataset into X(features) and y(labels: genre)
y = data['label']
X = data.loc[:, data.columns != 'label']

# SVM is a classifier that used for processing numeric data, out dataset class labels are string value, so encoder them into numeric one
label_encoder = LabelEncoder() # create label encoder object 
label_encoder.fit(y) # fit classes 
y = label_encoder.transform(y) # transform string into numeric value [blues = 0, jazz = 1, etc]

# split the dataset into training and validating set (8:2 for training:testing)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
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


# ======================================== Single Round Prediction ========================================
# Instantiate the KernelSVM object with a linear kernel and C=10
print("================")
print("SMO algorithm finding the optimal alpha with different kernels")
svm = KernelSVM(C=10, kernel_type="linear")
# Train the multi-class SVM model
svm.fit(X_train_pca, y_train)
# Make prediction using the trained multi-class Kernel SVM model
y_pred = svm.predict(X_test_pca)
# Calculate the accuracy of the model by comparing the predicted and true labels
accuracy = np.mean(y_test == y_pred)
# Print the accuracy
print("Accuracy with linear kernel:", accuracy)


# RBF Kernel
svm = KernelSVM(C=10, kernel_type="rbf", gamma=1/X_train_pca.shape[1])
svm.fit(X_train_pca, y_train)
y_pred = svm.predict(X_test_pca)
accuracy = np.mean(y_test == y_pred)
print(f"Accuracy with rbf kernel: {accuracy}")


# Laplace RBF Kernel
svm = KernelSVM(C=10, kernel_type="laplace_rbf", gamma=0.1)
svm.fit(X_train_pca, y_train)
y_pred = svm.predict(X_test_pca)
accuracy = np.mean(y_test == y_pred)
print("Accuracy with Laplace rbf kernel:", accuracy)

# Polynomial Kernel
# naming it gamma here, but actually we are inferring degree=2
svm = KernelSVM(C=10, kernel_type="polynomial", gamma=2)
svm.fit(X_train_pca, y_train)
y_pred = svm.predict(X_test_pca)
accuracy = np.mean(y_test == y_pred)
print("Accuracy with polynomial kernel:", accuracy)

# =================================== Cross Validation ==========================================
print("\n================")
# K fold with best kernel
C_values = [0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# Instantiate a KernelSVM object with a specific kernel
svm = KernelSVM(kernel_type="rbf", gamma=1/X_train_pca.shape[1])
# Find the best value of 'C' using cross-validation
best_C = None
best_accuracy = -1
# Iterate to find the best C
for C in C_values:
    accuracy = svm.cross_val_score(X_train_pca, y_train, C)
    print(f"Accuracy for C={C}: {accuracy}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_C = C
print(f"Best value of C: {best_C}")

# Train the final model with the best value of 'C'
# runtime
start_time = time.time()
svm.C = best_C
svm.fit(X_train_pca, y_train)
# Make prediction using the trained Kernal SVM model with the best value of 'C'
y_pred = svm.predict(X_test_pca)
accuracy = np.mean(y_test == y_pred)
end_time = time.time()
run_time = end_time - start_time 
print(f"Accuracy with best C: {accuracy}, runtime: {run_time}")


