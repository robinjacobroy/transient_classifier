import numpy as np
import pandas as pd
from pandas import Series , DataFrame
from astropy . time import Time # for time format conversions
import matplotlib as mpl
import matplotlib . pyplot as plt
from ipy_table import *
import FATS
get_ipython().magic('matplotlib inline')
mpl.style.use('ggplot') # Make the graphs a bit prettier
plt.rcParams ['figure.figsize'] = (15 , 5)
# ascii catalog of 47 ,000 sources
cv = pd.read_table('/Users/Linn/transient-classifier/data/CatalinaVars.tbl', delim_whitespace= True )

# Photometry data associated with the objects in the catalog
columns =['ID','MJD','Mag','Magerr','RA','Dec']
cp = pd.read_table('F:Down/AllVar.phot', header = None, names=columns, sep=',')
cp.head ()
# html table with key to vartypes
tb1 = pd.read_csv("C:/Users/Linn/transient-classifier/data/vartype.csv", index_col=0)
tb1.index +=1
tb1
# to cross match light curve data with variable types
# by merging photometric data with catalog and vartype data using two merges

mer = pd.merge(cp, cv.loc[:, ['Var_Type','Numerical_ID']], left_on='ID', right_on='Numerical_ID')
mer = pd.merge(mer, tb1.loc[:, ['Type']], left_on='Var_Type', right_index=True)
mer = mer.drop(['Numerical_ID','Var_Type'], axis =1)
mer.head ()
cd = mer.groupby(['ID']) # create a grouby object
u = pd.unique(mer['ID']) # find unique ids to loop over
# make light curves for some objects
for i in u [:26]:# loop over 10000 ids
  lt = cd.get_group(i)
  # lt.plot.scatter(x='MJD', y='Mag')
  t = lt['Type'].iloc[0]

  s = " LC of Type : " + t
  #plt.title(s)
  #plt.savefig("light curve of % i.png" %i)

#display statistics for a sample light curve
lt = cd.get_group(1109065026725)
lt.describe()
# make an error bar plot for an object
plt.errorbar(lt['MJD'] ,lt['Mag'] ,lt['Magerr'], fmt='ok')
# plot scatter matrix
pd.scatter_matrix(lt[['MJD','Mag','Magerr']], diagonal='kde')
print()
# plot of magnitudes against number
# lt . plot . scatter ( x ='MJD', y ='Mag')
lt.plot.line(y='Mag')
# feature extraction using FATS algorithm from http://isadoranun.github.io/tsfeat/ FeaturesDocumantation.html
a = FATS.FeatureSpace(Data =['magnitude','time','error'], featureList=['Amplitude','Autocor_length','Beyond1Std','MaxSlope','LinearTrend','MedianAbsDev','PercentAmplitude','Skew','Std','StetsonK','MedianBRP','SmallKurtosis'])
feat =[]# empty array to append feature array
for name ,group in cd:
  lc_sep = np.array([group['Mag'], group['MJD'], group['Magerr']])
  a = a.calculateFeature(lc_sep)
  r = a.result(method = 'dict')  # stores each feature set to a dict
  rf = pd.DataFrame.from_dict(r, orient = 'index')  # create df from dict
  rf = rf.T
  rf['Numerical_ID'] = name   # to merge on common column name
  feat.append(rf) # append the final df to array


# convert feat into dataframe
f = pd.concat(feat)
f = f.reset_index()
f = f.drop('index', axis=1)
# cross match feature table with ID and vartype using two merges
f = pd.merge(cv.loc[:, [ 'Numerical_ID', 'Var_Type']] ,f)
f = pd.merge(tb1.loc[:, ['Type ']], f, right_on='Var_Type', left_index=True)
f = f.drop(['Var_Type'], axis=1)
# write the table into file using
# f.to_csv("C:/Users/Linn/transient-classifier/data/feature_mat.csv")
feat = pd.read_csv("C:/Users/Linn/transient-classifier/data/feature_mat.csv")
feat.index +=1
feat = feat.drop('Unnamed : 0', axis =1)
feat.head ()
