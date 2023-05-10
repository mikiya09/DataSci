import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class momentumSVM:
    def __init__(self, kernel='linear', gamma=0.1, learning_rate=0.01, momentum=0.9, n_epochs=1000):
        self.kernel = kernel
        self.gamma = gamma
        self.lr = learning_rate
        self.momentum = momentum
        self.n_epochs = n_epochs
        self.alpha = None
        self.v = None
        self.W = None
        self.b = None

    # radio basis function kernel 
    def rbf_kernel(self, x, y):
        return np.exp(-self.gamma * np.sum((x - y) ** 2))

    # train the model 
    def fit(self, X_train, y_train):
        n_classes = len(np.unique(y_train))
        n_samples = X_train.shape[0]
        self.alpha = np.zeros((n_classes, n_samples))
        self.v = np.zeros((n_classes, n_samples))

        # Loop through all classes
        for c in range(n_classes):
            # Perform gradient ascent for n_epochs iterations
            for epoch in range(self.n_epochs):
                # Create binary labels for the current class
                y_train_binary = np.where(y_train == c, 1, -1)
                # Loop through all samples
                for i in range(n_samples):
                    # Calculate the kernel dot product based on the specified kernel
                    if self.kernel == 'linear':
                        kernel_dot_product = np.dot(y_train_binary * self.alpha[c], X_train @ X_train[i])
                    elif self.kernel == 'rbf':
                        kernel_dot_product = np.sum([self.alpha[c, j] * y_train_binary[j] * self.rbf_kernel(X_train[i], X_train[j]) for j in range(n_samples)])
                    else:
                        raise ValueError("Invalid kernel specified. Choose either 'linear' or 'rbf'.")

                    # Compute the gradient
                    gradient = 1 - y_train_binary[i] * kernel_dot_product
                    # Update the velocity vector
                    self.v[c, i] = self.momentum * self.v[c, i] + self.lr * gradient
                    # Update the Lagrange multipliers (alpha)
                    self.alpha[c, i] += self.v[c, i]

        # Calculate weights and biases from the Lagrange multipliers
        if self.kernel == 'linear':
            self.W = np.zeros((n_classes, X_train.shape[1]))
            for c in range(n_classes):
                y_train_binary = np.where(y_train == c, 1, -1)
                # Compute the weight vector for the linear kernel
                self.W[c] = np.dot(self.alpha[c] * y_train_binary, X_train)

        self.b = np.zeros(n_classes)
        for c in range(n_classes):
            y_train_binary = np.where(y_train == c, 1, -1)
            if self.kernel == 'linear':
                # Compute the bias for the linear kernel
                self.b[c] = np.mean(y_train_binary - np.dot(X_train, self.W[c]))
            elif self.kernel == 'rbf':
                # Compute the bias for the RBF kernel
                self.b[c] = np.mean(y_train_binary - np.array([np.sum([self.alpha[c, j] 
                                                      * y_train_binary[j] 
                                                      * self.rbf_kernel(X_train[i], X_train[j]) for j in range(n_samples)]) for i in range(n_samples)]))

    def predict(self, X_test, y_train):
        n_classes = len(np.unique(y_train))
        n_samples = X_test.shape[0]
        predictions = np.zeros((X_test.shape[0], n_classes))

        # Loop through all classes
        for c in range(n_classes):
            y_train_binary = np.where(y_train == c, 1, -1)
            # Loop through all test samples
            for i in range(X_test.shape[0]):
                # Calculate the kernel dot product based on the specified kernel
                if self.kernel == 'linear':
                    kernel_dot_product = np.dot(X_test[i], self.W[c])
                elif self.kernel == 'rbf':
                    kernel_dot_product = np.sum([self.alpha[c, j] 
                                                 * y_train_binary[j] 
                                                 * self.rbf_kernel(X_test[i], X_train[j]) for j in range(n_samples)])
                else:
                    raise ValueError("Invalid kernel specified. Choose either 'linear' or 'rbf'.")

                # Compute the prediction score for each class
                predictions[i, c] = kernel_dot_product + self.b[c]

        # Find the class with the highest prediction score for each test sample
        predicted_classes = np.argmax(predictions, axis=1)
        return predicted_classes


# Usage example
data = pd.read_csv("./features_30_sec.csv")

# Remove "filename" at the first column
data = data.iloc[:, 1:] 

y = data['label']
X = data.loc[:, data.columns != 'label']

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# Linear kernel
classifier_linear = momentumSVM(kernel='linear')
classifier_linear.fit(X_train, y_train)
predicted_classes_linear = classifier_linear.predict(X_test, y_train)
accuracy_linear = np.mean(predicted_classes_linear == y_test)
print(f"Linear kernel accuracy: {accuracy_linear}")

# RBF kernel
print("Momentum with radio basis function kernel take a long time to calculate accuracy, so I didn't train the model")
classifier_rbf = momentumSVM(kernel='rbf', gamma=0.1)
# classifier_rbf.fit(X_train, y_train)
# predicted_classes_rbf = classifier_rbf.predict(X_test, y_train)
# accuracy_rbf = np.mean(predicted_classes_rbf == y_test)
# print(f"RBF kernel accuracy: {accuracy_rbf}")


