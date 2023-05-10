
# Suppert Vector Macchine (Xirong Cao)


### Files 
*1) Dataset: [features_30_sec.csv](./features_30_sec.csv)* <br>
*2) [LinearSVM.py](./LinearSVM.py)* <br>
*3) [KernelSVM.py](./KernelSVM.py)* <br> 
*4) [PlotSVM.py](./PlotSVM.py)* <br> 
*5) [MomentumSVM.py](./MomentumSVM.py)*

### Usage
*• imported packages other than numpy, pandas, and matplotlib are only used for data preprocessing, the classifier is purely built with numpy. Runtime is calculated automatically for best improvement in each file.* <br>
```shell
# run any .py file will print out all the experiemnts
>> python LinearSVM.py KernelSVM.py PlotSVM.py
```

#### 1) Dataset
*dataset is 30s music clips with different genre, and it needs be kept in the same directory with .py file because of the way I load it.*
```python
import pandas
data = pd.read_csv("./features_30_sec.csv")
```
#### 2) LinearSVM.py
*linear SVM classifier with pure hinge loss accuracy calculatied, three different slack variables as regularization term, PCA method removed accuracy, OvO and OvR implementation*
```python
# create SVM class instance 
svm = LinearSVM()
# fit the model with test feature and encoded y label
svm.fit(X_train_pca, y_train)
# make prediction using predict_ovr method 
prediction = svm.predict_ovr(X_test_pca)
# print the accuracy score 
print("SVM without regularization, accuracy: ", accuracy_score(y_test, prediction))
```
*`method introduction:`* <br>
```python
# train customized slack variable w^2
svm.fit_ovr_L(X_train_pca, y_train)

# train L1
svm.fit_ovr_L1(X_train_pca, y_train)

# train L2
svm.fit_ovr_L2(X_train_pca, y_train)

# One-vs-One method 
svm.fit_ovo_L2(X_train_pca, y_train)
```

#### 3) KernelSVM.py
*Kernelization SVM solve the dual problem with Sequential minimal optimization (SMO) algorithm, four kernel choices: linear, polynomial, rbf, laplace_rbf. Cross validation with 5 folds.*
```python
# create KernelSVM instance
svm = KernelSVM(C=10, kernel_type="linear")
# Train the SVM model
svm.fit(X_train_pca, y_train)
# Make prediction 
y_pred = svm.predict(X_test_pca)
# Calculate the accuracy of the model by comparing the predicted and true labels
accuracy = np.mean(y_test == y_pred)
# Print the accuracy
print("Accuracy with linear kernel:", accuracy)

# # RBF Kernel
svm = KernelSVM(C=10, kernel_type="rbf", gamma=1/X_train_pca.shape[1])
svm.fit(X_train_pca, y_train)
y_pred = svm.predict(X_test_pca)
accuracy = np.mean(y_test == y_pred)
print("Accuracy with rbf kernel:", accuracy)

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

# print optimal alpha value for one KernelSVM instance 
for pair, (support_vectors, support_alphas, support_labels, b) in svm.classifiers.items():
    print(f"Alpha values for classifier {pair}: \n{support_alphas}")

# cross validation with a set of C value 
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
svm.C = best_C
svm.fit(X_train_pca, y_train)

# Make prediction using the trained multi-class SVM model with the best value of 'C'
y_pred = svm.predict(X_test_pca)
accuracy = np.mean(y_test == y_pred)
print(f"Accuracy with best C: {accuracy}")
```
#### 4) PlotSVM.py
*didn't have time to implement function that directly collects accuracy value from previous .py file, all data for comapred accuracy is manually entered.*
```shell
>> python PlotSVM.py
```
#### 5) MomentumSVM.py
*this improvement perform worse than baseline, so didn't pay lots of attention to it, but still can be run with the following command*
```shell
>> python MomentumSVM.py
```

