import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from astropy.time import Time
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.neighbors import KNeighborsClassifier
get_ipython().magic('matplotlib inline')
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import svm
from sklearn.cross_validation import train_test_split
# read feature table
data = pd.read_csv("C:/Users/Linn/transient-classifier/data/feature_mat.csv")
a = data.columns
data = data.drop(labels = a[[0 ,2]] ,axis =1)
colors = [ 'r', 'b', 'g', 'c', 'm ']
markers = [ 'd', 's', 'o', 'v', '^', 'p', 'd', 'x', '+', 'h', 's', '8']
# change this array to plot different features and types
feat =['StetsonK','SmallKurtosis']
types =['Blazkho','HADS','EA_UP','ELL']
colors = ['r', 'g', 'b', 'm']
# loop to scatter plot features of two types at a time

for l, c, m in zip(types ,colors ,markers[:4]) :
  d = data [data['Type']== l] # selects data for each of the types
  #plt.scatter(x = data['Amplitude'][c1], y=data['PercentAmplitude'][c1])
  plt.scatter(x =d[feat[0]], y = d[feat[1]], c=c, label=l)
plt.ylim(-10 ,50)
plt.xlabel(feat[0])
plt.ylabel(feat[1])
plt.legend(loc='upper right ')
plt.savefig(feat[0]+"_"+ feat[1]+ str(types) +".png", dpi=600)
# obtain feature matrix for implementing PCA
X = data.drop(labels=['Type','MaxSlope'], axis =1)
y = np.array(data['Type'])
# convert y into numpy array for scatter plot to work
pca = PCA()
plt.semilogy(pca.fit(X).explained_variance_ratio_, '--o')
pca = PCA(n_components=5)
reduced_X = pca.fit_transform(X)
types = ['Blazkho','HADS','EA_UP','ELL']
feat = ['Amplitude','StetsonK']
colors = ['r', 'g', 'b', 'm']
# loop to scatter plot PCA components
for l ,c , m in zip(types , colors , markers[:4]):
  plt.scatter(reduced_X[y == l, 4], \
              reduced_X[y == l, 3], c=c, marker=m, label = l ) # plot first and second component of each type
plt.xlabel('PC 4')
plt.ylabel('PC 3')
plt.legend(loc='upper right')
plt.savefig ("3-4"+str(types) +".png")
# implementing SVM
X = data.loc[data['Type'].isin (['EA_UP','Cep-II','ELL','ACEP','PCEB'])]
y = X['Type']
X = X.drop(labels=['Type','MaxSlope'], axis=1)
pca = PCA(n_components =5)
reduced_X = pca.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(reduced_X, y, test_size=0.4)
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)
expected = y_test
predicted = model.predict(X_test)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))                     
