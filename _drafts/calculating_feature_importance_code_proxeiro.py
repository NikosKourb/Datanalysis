import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder #Για unordered categorical features
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report,confusion_matrix,precision_recall_fscore_support
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

from matplotlib import pyplot

import time
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

from classes.Column import Column


from functions.models.create_train_predict_analyze import create_model
from functions.calculating_feature_importance import feature_importance



'''Calculate Feature Importance'''

rel_path = 'files/csv/'

feature_importance(rel_path + 'Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv',
                   ['REGISTRATION_ID','LAB_RESULT','FINAL_CLASSIFICATION'],'SURVIVED',
                   'lgr',{},22)

feature_importance(rel_path + 'Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv',
                   ['REGISTRATION_ID','LAB_RESULT','FINAL_CLASSIFICATION'],'SURVIVED',
                   'dtc',{},22)

feature_importance(rel_path + 'Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv',
                   ['REGISTRATION_ID','LAB_RESULT','FINAL_CLASSIFICATION'],'SURVIVED',
                   'dtr',{},22)

feature_importance(rel_path + 'Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv',
                   ['REGISTRATION_ID','LAB_RESULT','FINAL_CLASSIFICATION'],'SURVIVED',
                   'rfc',{},22)

feature_importance(rel_path + 'Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv',
                   ['REGISTRATION_ID','LAB_RESULT','FINAL_CLASSIFICATION'],'SURVIVED',
                   'rfr',{},22)

feature_importance(rel_path + 'Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv',
                   ['REGISTRATION_ID','LAB_RESULT','FINAL_CLASSIFICATION'],'SURVIVED',
                   'xbc',{},22)

feature_importance(rel_path + 'Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv',
                   ['REGISTRATION_ID','LAB_RESULT','FINAL_CLASSIFICATION'],'SURVIVED',
                   'xbr',{},22)

feature_importance(rel_path + 'Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv',
                   ['REGISTRATION_ID','LAB_RESULT','FINAL_CLASSIFICATION'],'SURVIVED',
                   'knc',{},22)

feature_importance(rel_path + 'Covid19MPD_8_23_en_pos_fc_valid_lb_none.csv',
                   ['REGISTRATION_ID','LAB_RESULT','FINAL_CLASSIFICATION'],'SURVIVED',
                   'knr',{},22)


'''
#Calculate feature importance for Random Forest
def feature_importance_r_forest(rel_path,df_file_name,rand_state):
    
    dataframe = pd.read_csv(rel_path + df_file_name)
    
    model=RandomForestClassifier(random_state=rand_state)
    
    features=dataframe.drop(columns=['REGISTRATION_ID','SURVIVED'])
    
    model.fit(features,dataframe['SURVIVED'])
 
    feature_importances=pd.DataFrame({'features':features.columns,'feature_importance':model.feature_importances_})
    feature_importances.sort_values('feature_importance',ascending=False)
    
    print(feature_importances)
'''

''' 
#load the dataset
def load_dataset(filename,x_col_num_list,y_col_num_list):
	# load the dataset as a pandas DataFrame
	data = pd.read_csv(filename, header=None)
	# retrieve numpy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
    y = dataset[:,-1]
	#X = dataset[x_col_num_list]
    #y = dataset[y_col_num_list]
	# format all fields as string
	X = X.astype(str)
	return X, y
'''

'''
# load the dataset
#def load_dataset(filename,x_col_num_list,y_col_num):
def load_dataset(filename,x_col_num_list,y_col_num):
	# load the dataset as a pandas DataFrame
	data = pd.read_csv(filename, header=None)
	# retrieve numpy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:,x_col_num_list]
	y = dataset[:,y_col_num]
	# format all fields as string
	X = X.astype(str);y = y.astype(str)
	return X, y

 
# prepare input data
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
	X_train_enc = oe.transform(X_train)
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc
 
# prepare target
def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc
 
# feature selection
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=mutual_info_classif, k='all')
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs


dataset = pd.read_csv('01.Covid19MPD.(Pos_valid_bin_21).csv')
# dataset2 = pd.read_csv('covid_jpn_total.csv')
# dataset3 = pd.read_csv('all-states-history.csv')

'''''''
Μία γρήγορη ματιά στο dataset με functions των pandas
'''''''
print(dataset.head())
print(dataset.info())
print(dataset.describe())
    
#scale=StandardScaler()
#X_train = scale.fit_transform(X_train)
#X_test = scale.transform(X_test)

# load the dataset except the columns 06.AGE,20.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION
#X, y = load_dataset('01.Covid19MPD.(Pos_transf_bin_21).csv',[1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18],20)
X, y = load_dataset('01.Covid19MPD.(Pos_valid_bin_21).csv',[1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18],20)
print(X , '\n' + '\n', y)

# split into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,train_size=0.8)
#get shape of train and test data

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
#X_train, X_test, y_train, y_test = train_test_split(dataset.loc[:, dataset.columns != 'class'], dataset['class'], test_size=0.2, random_state=42)

# summarize
print("train size X : ",X_train.shape)
print("train size y : ",y_train.shape)
print("test size X : ",X_test.shape)
print("test size y : ",y_test.shape)
#print('Train', X_train.shape, y_train.shape)
#print('Test', X_test.shape, y_test.shape)

# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)

# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)

# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))

# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()
'''