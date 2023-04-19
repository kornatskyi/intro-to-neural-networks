from sklearn.datasets import load_iris
import random
import numpy as np
random.seed(42)

iris = load_iris()

def normalize(data: list):
  min_values = np.min(data, axis=0)
  max_values = np.max(data, axis=0)
  return (data - min_values) / (max_values - min_values)

combined_data = list(zip(normalize(iris.data),normalize(iris.target)))
random.shuffle(combined_data)
shx, shy = zip(*combined_data)  
X, y = list([list(x) for x in shx]), list(shy)




training_data_percent = 0.80
training_size = int(len(X)*training_data_percent)
testing_size = len(X) - training_size
training_data_points, training_targets = X[:training_size],y[:training_size]   
testing_data_points, testing_targets = X[-testing_size:], y[-testing_size:]   

