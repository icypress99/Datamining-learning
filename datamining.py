import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


# Read the dataset
df = pd.read_csv('liver_cirrhosis.csv')

label_encoder = LabelEncoder()
df['Status'] = label_encoder.fit_transform(df['Status'])
df['Drug'] = label_encoder.fit_transform(df['Drug'])
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Ascites'] = label_encoder.fit_transform(df['Ascites'])
df['Hepatomegaly'] = label_encoder.fit_transform(df['Hepatomegaly'])
df['Spiders'] = label_encoder.fit_transform(df['Spiders'])
df['Edema'] = label_encoder.fit_transform(df['Edema'])

df_encoded = pd.DataFrame(df)
X = df_encoded.drop('Stage', axis=1)  
y = df_encoded['Stage'] 

k = 5

kf = KFold(n_splits=k)

rf_accuracies = 0
rf_recalls = 0
rf_f1 = 0

naive_bayes_accuracies = 0
naive_bayes_recalls = 0
naive_bayes_f1 = 0

knn_accuracies = 0
knn_recalls = 0
knn_f1 = 0

svm_accuracies = 0
svm_recalls = 0
svm_f1 = 0

dt_accuracies = 0
dt_recalls = 0
dt_f1 = 0

# Split the data into training and testing sets
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


    # Logistic Regression:
    # logreg = LogisticRegression()
    # logreg.fit(X_train, y_train)
    # accuracy = logreg.score(X_test, y_test)
    # print(" Logistic Regression Accuracy:", accuracy)
    
    # Random Forest:
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_accuracies += rf.score(X_test, y_test)

    # Random Forest
    rf_pred = rf.predict(X_test)
    rf_cm = confusion_matrix(y_test, rf_pred)
    print("Random Forest Confusion Matrix:")
    print(rf_cm)

    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, rf_pred)
    rf_recalls += recall.mean()
    rf_f1 += f1_score.mean()
    
    
    # Naïve Bayes:
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)
    naive_bayes_accuracies += naive_bayes.score(X_test, y_test)

    naive_bayes_pred = naive_bayes.predict(X_test)
    naive_bayes_cm = confusion_matrix(y_test, rf_pred)
    print("Naïve Bayes Confusion Matrix:")
    print(naive_bayes_cm)

    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, naive_bayes_pred)
    naive_bayes_recalls += recall.mean()
    naive_bayes_f1 += f1_score.mean()

    # K-Nearest Neighbors (KNN) :
    knn = KNeighborsClassifier(metric='euclidean', n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_accuracies += knn.score(X_test, y_test)
    
    knn_pred = knn.predict(X_test)
    knn_cm = confusion_matrix(y_test, knn_pred)
    print("KNN Confusion Matrix:")
    print(rf_cm)
    
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, knn_pred)
    knn_recalls += recall.mean()
    knn_f1 += f1_score.mean()

    #SVM:
    svm = SVC(kernel='rbf')
    svm.fit(X_train, y_train)
    svm_accuracies += svm.score(X_test, y_test)
     
    svm_pred = svm.predict(X_test)
    svm_cm = confusion_matrix(y_test, svm_pred)
    print("SVM Confusion Matrix:")
    print(rf_cm)
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, svm_pred)
    svm_recalls += recall.mean()
    svm_f1 += f1_score.mean()
    
    # Decision Tree
    decision_tree = DecisionTreeClassifier(max_depth=5)
    decision_tree.fit(X_train, y_train)
    dt_accuracies = decision_tree.score(X_test, y_test)
    
    dt_pred = decision_tree.predict(X_test)
    dt_cm = confusion_matrix(y_test, dt_pred)
    print("decision_tree Confusion Matrix:")
    print(rf_cm)

    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, dt_pred)
    dt_recalls += recall.mean()
    dt_f1 += f1_score.mean()

    print("---------------------------------------------" )
    
rf_accuracies = rf_accuracies / k
rf_recalls = rf_recalls / k
rf_f1 = rf_f1 / k

naive_bayes_accuracies = naive_bayes_accuracies / k
naive_bayes_recalls = naive_bayes_recalls / k
naive_bayes_f1 = naive_bayes_f1 / k

knn_accuracies = knn_accuracies / k
knn_recalls = knn_recalls / k
knn_f1 = knn_f1 / k

svm_accuracies = svm_accuracies / k
svm_recalls = svm_recalls / k
svm_f1 = svm_f1 / k

dt_accuracies = dt_accuracies / k
dt_recalls = dt_recalls / k
dt_f1 = dt_f1 / k

accuracies_models = {
    rf_accuracies: "Random Forest",
    naive_bayes_accuracies: "Naive Bayes",
    knn_accuracies: "KNN Accuracy",
    svm_accuracies: "SVM Accuracy",
    dt_accuracies: "Decision Tree"
}

accuracies_models = sorted(accuracies_models.items(), key=lambda x: x[0], reverse=True)
best_model = accuracies_models[0]
best_accuracies, best_model_name = best_model
print(f"Best model based on Accuracy name: {best_model_name}, Value: {best_accuracies}")


recall_models = {
    rf_recalls: "Random Forest",
    naive_bayes_recalls: "Naive Bayes",
    knn_recalls: "KNN Accuracy",
    svm_recalls: "SVM Accuracy",
    dt_recalls: "Decision Tree"
}

recall_models = sorted(recall_models.items(), key=lambda x: x[0], reverse=True)
best_model = recall_models[0]
best_accuracies, best_model_name = best_model
print(f"Best model based on Recall name: {best_model_name}, value: {best_accuracies}")



f1_models = {
    rf_f1: "Random Forest",
    naive_bayes_f1: "Naive Bayes",
    knn_f1: "KNN Accuracy",
    svm_f1: "SVM Accuracy",
    dt_f1: "Decision Tree"
}

f1_models = sorted(f1_models.items(), key=lambda x: x[0], reverse=True)
best_model = recall_models[0]
best_accuracies, best_model_name = best_model
print(f"Best model based on F1-score name: {best_model_name}, value: {best_accuracies}")

# for accuracies, model_name in accuracies_models:
#     print(f"Model: {model_name}, Accuracies: {accuracies}")


print("******************************")



