# Import the packages
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import linear_model
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Fetch dataset 
dry_bean_dataset = fetch_ucirepo(id=602) 

# Data (as pandas dataframes) 
X = dry_bean_dataset.data.features 
y = dry_bean_dataset.data.targets 

# Count number of different bean varieties
print(y['Class'].value_counts())

# Plot bar chart
sns.countplot(x='Class', data=y).set(title='Class Distribution')


# Split the data into training and testing
xTrain, xTest, yTrain, yTest = train_test_split(X, np.ravel(y), test_size=0.2, stratify=y, random_state=10)
# Scale the data
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

# Store accuracy score of different classifiers
acc_score = []
classifiers = ['Decision Tree', 'Nearest Neighbour', 'Naïve Bayes', 'SVM', 'Logistic Regression']

# Create Labels for confusion matrix
labels = ['Seker', 'Barbunya', 'Bombay', 'Cali', 'Dermosan', 'Horoz', 'Sira']

# 1) Decision Tree
# Find the ideal depth for the decision tree
training_acc = []
testing_acc = []
depths = range(1, 20)
    
    
for depth in depths:
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    # training
    clf.fit(xTrain, yTrain)
    # predicting
    trainPred = clf.predict(xTrain)
    testPred = clf.predict(xTest)
    # computing and saving accuracy 
    training_acc.append(accuracy_score(yTrain, trainPred))
    testing_acc.append(accuracy_score(yTest, testPred))


# Ploting the training and test accuracies vs the tree depths  
plt.figure()
plt.plot(depths, training_acc,'rv-',depths, testing_acc,'bo--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Tree Depth')
plt.ylabel('Classifier Accuracy')
plt.title("Decision Tree Depth")
plt.show()

decision_acc = round(np.max(testing_acc)*100, 1)
acc_score.append(decision_acc)

# Printing the max testing accuracy score 
print('The testing accuracy for decision tree classifier is', decision_acc, '% when tree depth is', np.argmax(testing_acc)+1)

# Plot tree with max depth of 11(may vary very slightly)
clf = tree.DecisionTreeClassifier(max_depth=11)
clf.fit(xTrain, yTrain)
plt.figure(figsize=(15,10))
tree.plot_tree(clf, fontsize=12)
plt.show()


# 2) Nearest-neighbor

# Find optimal K
training_acc = []
testing_acc = []
k_range = range(1, 11)
    
    
for i in k_range:
    clf = KNeighborsClassifier(n_neighbors = i)
    # training
    clf.fit(xTrain, yTrain)
    # predicting
    trainPred = clf.predict(xTrain)
    testPred = clf.predict(xTest)
    # computing and saving accuracy 
    training_acc.append(accuracy_score(yTrain, trainPred))
    testing_acc.append(accuracy_score(yTest, testPred))

# Ploting the training and test accuracies vs the tree depths  
plt.figure()
plt.plot(k_range, training_acc,'rv-',k_range, testing_acc,'bo--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('K')
plt.ylabel('Classifier Accuracy')
plt.title("KNN")

knn_acc = round(np.max(testing_acc)*100, 1)
acc_score.append(knn_acc)

print('The testing accuracy for KNN classifier is', knn_acc, '% when K is', np.argmax(testing_acc)+1)


# 3) Naïve Bayes
# Build and train a Gaussian Classifier
clf = GaussianNB()
clf.fit(xTrain, yTrain)


# Predict and print
yPred= clf.predict(xTest)
knn_acc = round(accuracy_score(yTest, yPred)*100, 1)
acc_score.append(knn_acc)
print("The accuracy for Naïve Bayes classifier is", knn_acc , "%.")
# Plot the confusion matrix
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.title("Naïve Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True Label")

# 4) Support Vector Machine
# Create and train a svm classifier
clf = svm.SVC(kernel='linear') 
clf.fit(xTrain, yTrain)

# Predict
yPred = clf.predict(xTest)
svm_acc = round(accuracy_score(yTest, yPred)*100, 1)
acc_score.append(svm_acc)
print("The accuracy for SVM classifier is", svm_acc, "%.")

# Plot the confusion matrix
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", xticklabels=labels, yticklabels=labels)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True Label")


# 5) Logistic Regression
# Build and train logic model
reg = linear_model.LogisticRegression(multi_class='multinomial', max_iter=1000) 
reg.fit(xTrain, yTrain)

# Predict
yPred = reg.predict(xTest) 
reg_acc = round(accuracy_score(yTest, yPred)*100, 1)
acc_score.append(reg_acc)
print("The accuracy of Logistic Regression is", reg_acc, "%.")

# Plot the confusion matrix
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True Label")

# Plot the accuracy scores
plt.figure()
plt.bar(classifiers, acc_score)
plt.xticks(rotation=45) 
plt.xlabel('Classifiers')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores of 5 Classifiers')
plt.ylim(80, 100)  #set y-axis range