import sys
import pandas as pd
import numpy as np
import PEASUtil
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import joblib
import argparse
import os
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance

########################DEEPINSIGHT
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
import inspect
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
###########################################

wd = os.getcwd()

#argument parsing
parser = argparse.ArgumentParser(description='Trains a multi-layer perceptron neural network model for ATAC-seq peak data.')
parser.add_argument('featurefiles', type=str, help='File listing the file paths of all features to train the model.')

parser.add_argument('-o', dest='out', type=str, help='The selected directory saving outputfiles.')
parser.add_argument('-n', dest='name', type=str, help='Name of the Model.')
parser.add_argument('-p', dest='paramstring', help='String containing the parameters for the model.', type=str)
parser.add_argument('-f', dest='features', help='Feature index file specifying which columns to include in the feature matrix.', type=str)
parser.add_argument('-c', dest='classes', help='File containing class label transformations into integer representations.', type=str)
parser.add_argument('-l', dest='labelencoder', help='File containing feature label transformations into integer representations.', type=str)
parser.add_argument('-r', dest='randomstate', help='Integer for setting the random number generator seed.', type=int, default=929)

args = parser.parse_args()

#Required Arguments
datasetlabels, datasetfiles = PEASUtil.getDatasets(args.featurefiles)

#Optional Arguments
featurefiledirectory = os.path.dirname(args.featurefiles)
featurefilename = os.path.splitext(os.path.basename(args.featurefiles))[0]

if args.name is not None:
    modelname = args.name
    modelnamefile = args.name.replace(" ", "_")
else:
    modelname = featurefilename
    modelnamefile = featurefilename.replace(" ", "_")

if args.out is not None:
    outdir = PEASUtil.getFormattedDirectory(args.out)
else:
    outdir = PEASUtil.getFormattedDirectory(featurefiledirectory)


parameters = PEASUtil.getModelParameters(args.paramstring)

if args.features is not None:
    featurecolumns = PEASUtil.getFeatureColumnData(args.features)
else:
    featurecolumns = PEASUtil.getFeatureColumnData(wd+"/features.txt")

if args.classes is not None:
    classconversion = PEASUtil.getClassConversions(args.classes)
else:
    classconversion = PEASUtil.getClassConversions(wd+"/classes.txt")

if args.labelencoder is not None:
    labelencoder = PEASUtil.getLabelEncoder(args.labelencoder)
else:
    labelencoder = PEASUtil.getLabelEncoder(wd+"/labelencoder.txt")

randomstate = args.randomstate
parameters['random_state'] = randomstate

#Model Training
#imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = SimpleImputer(strategy='mean')
trainX = np.zeros((0,len(featurecolumns)))
trainy = np.zeros((0,))
print("Reading feature files")
for curfile in datasetfiles:
    curdata = pd.read_csv(curfile, sep="\t")
    trainXi, trainyi, _, _, = PEASUtil.getData(curdata, featurecolumns, labelencoder, classconversion)
    trainXi = preprocessing.StandardScaler().fit_transform(imputer.fit_transform(trainXi))
    trainX = np.concatenate((trainX, trainXi))
    trainy = np.concatenate((trainy, trainyi))

train_X,test_X,train_y,test_y = train_test_split(trainX,trainy,test_size=0.2,random_state=5)
#mlp_clf__tuned_parameters = {"hidden_layer_sizes": [(25,),(50,),(100,50),(200,100),(100,25),(200,)],
#                             "activation":['relu','tanh','logistic'],
##                             "solver": ['adam'],
#                             "verbose": [True],
#                             "beta_1":[0.999,0.8],
#                             "beta_2":[0.9999,0.999,0.8],
#                             "epsilon":[1e-08,1e-06,1e-10]
#                             }
########################################Deepinsight




########################################
#SVM_parameters = {'kernel': ['rbf'], 
#                  'gamma': [1e-3, 1e-2,1e-4],
#                     'C': [1, 10, 100, 1000],
 #                 "verbose": [True]
 #                }
                    

#mlp = MLPClassifier()
#SVM = SVC(probability=True)
#clf = GradientBoostingClassifier()

#print('searching best..')
#clf = GridSearchCV(SVM, SVM_parameters, n_jobs=5)
print("Training Model")
#clf = SVC(probability=True)
#clf = RandomForestClassifier(random_state=0)
#clf = KNeighborsClassifier(n_neighbors=20)
#clf = MLPClassifier(**parameters)
#最优#
clf=MLPClassifier(solver='adam',beta_1=0.999,beta_2=0.999,epsilon=0.000001,activation='logistic',hidden_layer_sizes=(200,))
print(clf)
clf.fit(trainX, trainy)
#print("Best",clf.best_params_)

####################get feature inportance
#clf=MLPClassifier(solver='adam',beta_1=0.999,beta_2=0.999,epsilon=0.000001,activation='logistic',hidden_layer_sizes=(200,))
#print(clf)
#clf.fit(trainX,trainy)
#results = permutation_importance(clf, trainX, trainy, scoring='accuracy')
 #get importance
#importance = results.importances_mean

#import matplotlib.pyplot as plt
#summarize feature importance
#for i,v in enumerate(importance):
#    print('Feature: %s, Score: %.5f' % (i,v))

#plot feature importance
#plt.bar([x for x in range(len(importance))], importance)
#plt.savefig('featureInportance.jpg')
#plt.show()
#output = cross_validate(clf, trainX,trainy, cv=5, scoring = 'accuracy',return_estimator =True)
#print(output)
####################################################


outfile = outdir+modelnamefile+'.pkl'
print("Writing model to: "+outfile)
joblib.dump(clf, outfile)
print("Complete.")