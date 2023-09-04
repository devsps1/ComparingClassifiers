import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
import timeit

def PrintConfusionMatrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

def PrintClassificationReport(y_test, y_pred):
   classification_rep = classification_report(y_test, y_pred)
   print("Classification Report:")
   print(classification_rep)

if __name__ == "__main__":
  banknote_data = pd.read_csv("data_banknote.csv", header=None)

  X = banknote_data.iloc[:, :-1] # All columns except the last one
  y = banknote_data.iloc[:, -1] # Last column

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

  print(f'Train: {X_train.shape}')
  print(f'Test: {X_test.shape}')

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  clf = LinearDiscriminantAnalysis()
  svm_classifier = SVC(kernel='linear')

  #--------------------------------------- Fisher Linear Discriminant---------------------------------------
  # training time
  print("\n\t\tFisher's Linear Discriminant\n")
  start_time = timeit.default_timer()
  clf.fit(X_train_scaled, y_train)
  finish_time = timeit.default_timer()
  training_time = finish_time - start_time
  print(f"Training Time: {training_time:.5f} seconds")

  # testing time
  start_time = timeit.default_timer()
  y_pred = clf.predict(X_test_scaled)
  finish_time = timeit.default_timer()
  testing_time = finish_time - start_time
  print(f"Testing Time: {testing_time:.5f} seconds")

  PrintConfusionMatrix(y_test, y_pred)
  PrintClassificationReport(y_test, y_pred)

  # --------------------------------------- Linear support vector machine------------------------------------
  # training time
  print("\n\n\t\tSupport Vector Machine\n")
  start_time = timeit.default_timer()
  svm_classifier.fit(X_train_scaled, y_train)
  finish_time = timeit.default_timer()
  training_time = finish_time - start_time
  print(f"Training Time: {training_time:.5f} seconds")

  # testing time
  start_time = timeit.default_timer()
  y_pred = svm_classifier.predict(X_test_scaled)
  finish_time = timeit.default_timer()
  testing_time = finish_time - start_time
  print(f"Testing Time: {testing_time:.5f} seconds")

  PrintConfusionMatrix(y_test, y_pred)
  PrintClassificationReport(y_test, y_pred)