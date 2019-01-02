import numpy as np
from urllib.request import urlopen

data = np.loadtxt(urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"),delimiter=",")
X = data[:,0:8]
y = data [:,8]