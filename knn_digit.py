from __future__ import division
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA

def init():
  global DATA_DIR, DATA_FILENAME, data_file
  global HOME_DIR

  # cache
  _join = os.path.join

  # system home directory
  HOME_DIR = os.getenv('HOME')

  DATA_DIR = _join(HOME_DIR, 'Downloads/datasets/uci-digits')
  DATA_FILENAME   = 'semeion.data'

  data_file    = _join(DATA_DIR, DATA_FILENAME)

def prepare_data(test_size=0.2):
  '''
  test_size - fraction of the dataset to be used for testing

  (1-test_size) fraction of the data will be used for training
  '''
  global x_train, y_train, x_test, y_test
  global data_file

  features, labels  = [], []

  with open(data_file) as fh:
      for line in fh:
          tokens = line.split()
          # first 256 tokens are features
          # next 10 tokens are labels, boolean values, 1 for the digit and 0 for others
          feature, label = map(float, tokens[:256]), str(np.where(np.array(map(int, tokens[256:266]))>0)[0][0])
          
          features.append(feature)
          labels.append(label)
          
  x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)

  x_train = np.array(x_train)
  x_test = np.array(x_test)

def visualize_data():
  global x_train, y_train

  # PCA using two components to visualize the data in 2-D
  pca = RandomizedPCA(n_components=2)
  X = pca.fit_transform(x_train)
  df = pd.DataFrame({'x' : X[:, 0], 'y' : X[:, 1], 'label' : y_train})

  colors = ['r', 'g', 'b', 'c', 'm', 'y', '#EEEEEE' , '#888888', '#00DD44', '#DDFF22']
  for label, color in zip(df['label'].unique(), colors):
      mask = df['label'] == label
      plt.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
  plt.legend()
  plt.show()

def train_knn():
  clf = KNN(n_neighbors=5)
  clf.fit(x_train, y_train)

  return clf

def evaluate(classifier, which = 'KNN-Classifier'):
  global x_test, y_test

  prediction = classifier.predict(x_test)
  y_test_int = np.array(map(int, y_test))
  y_pred = np.array(map(int, prediction))
  diff = y_test_int - y_pred

  print 'Accuracy of ', which, ' is : ', (len(y_test) - len(diff[diff!=0]))/len(y_test)
  print 'Below is the confusion matrix: '
  print confusion_matrix(y_test, prediction)

def main():
  init()
  prepare_data()
  # visualize_data()
  knn_classifier = train_knn()
  # predict(knn_classifier)
  evaluate(knn_classifier)

if __name__ == '__main__':
  main()