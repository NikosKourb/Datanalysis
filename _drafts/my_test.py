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
from sklearn import preprocessing
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

import time
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

from Column import Column

from my_functions import feature_correlation_std
from my_functions import prepare_dataset
from my_functions import model_create_train_pred_analysis
from my_functions import find_model_opt_param
from my_functions import create_multi_df_categorical_column_list_chart,create_multi_df_categorical_column_bar_chart,create_multi_df_numerical_column_bar_chart
from my_functions import create_model,classification_report_opt_metrics
from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

''''''
from sklearn.utils import shuffle


'''

covid19_df = pd.read_csv('01.Covid19MPD.(Pos_valid_bin_21).csv')
print(covid19_df.head())
print(covid19_df.info())
print(covid19_df.describe())

scaler = StandardScaler()
ros = RandomOverSampler()
scaler = MinMaxScaler()

Pipeline:
pipe = Pipeline(steps=[('RF', rand_forest_clf)], memory='tmp')

'''

#Create a barchart from a dataframe list of columns with categorical values as an addage to the above function
def create_multi_df_categorical_column_list_chart(df,col_dict,chart_title):
    my_column_values_list=['yes','no','not apply','ignored','not specified']
    dict_common ={'yes':'y','no':'n','not apply':'n|a'}
    my_col_list=[]
    
    for col in col_dict:
        my_col_list.append(Column(str(col),str(col_dict[col]),dict_common))
    
    create_multi_df_categorical_column_bar_chart(df,my_col_list,my_column_values_list,chart_title)
    
    
    
#Create Ages & Sexes Barchart
def ages_column_barchart(df,col_ages_name,col_sexes_name,chart_title):
    column_sexes_values_list= ['women','men','not apply','ignored','not specified']
    column_ages_values_list= ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99','100 and over']
    
    sexes_column_list =[Column(col_sexes_name,'Sexes',{'women':1,'men':0,'not apply':97,'ignored':98,'not specified':99})]
    ages_column_list =[Column(col_ages_name,'0-9',{'0-9':[0,9]}),Column(col_ages_name,'10-19',{'10-19':[10,19]}),
                       Column(col_ages_name,'20-29',{'20-29':[20,29]}),Column(col_ages_name,'30-39',{'30-39':[30,39]}),
                       Column(col_ages_name,'40-49',{'40-49':[40,49]}),Column(col_ages_name,'50-59',{'50-59':[50,59]}),
                       Column(col_ages_name,'60-69',{'60-69':[60,69]}),Column(col_ages_name,'70-79',{'70-79':[70,79]}),
                       Column(col_ages_name,'80-89',{'80-89':[80,89]}),Column(col_ages_name,'90-99',{'90-99':[90,99]}),
                       Column(col_ages_name,'100 and over',{'100 and over':[100,500]})]
    
    create_multi_df_numerical_column_bar_chart(df,ages_column_list,column_ages_values_list,sexes_column_list,column_sexes_values_list,chart_title)



def preprocessing_dataframe(df_csv_file,scaler_name):
    
    if  scaler_name == 'std':
        scaler = StandardScaler()
    
    elif scaler_name == 'mm':
        scaler = MinMaxScaler(feature_range=(0,1))
    
    le = LabelEncoder()
    
    dataframe = pd.read_csv(df_csv_file)
    
    dataframe[['06.AGE']] = scaler.fit_transform(dataframe[['06.AGE']])
    
    dataframe[['20.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION']] = scaler.fit_transform(dataframe[['20.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION']])
    
    dataframe = pd.get_dummies(dataframe, columns=['07.PREGNANCY'])
    
    #dataframe = dataframe.drop(columns=['01.REGISTRATION_ID','03.OUTPATIENT'])
    
    col_list = ['02.SEX','04.INTUBATED','05.PNEUMONIA','08.DIABETIC','09.COPD','10.ASTHMA',
                   '11.IMMUNOSUPPRESSED','12.HYPERTENSION','13.OTHER_CHRONIC_DISEASE','14.CARDIOVASCULAR',
                   '15.OBESITY','16.CHRONIC_KIDNEY_FAILURE','17.SMOKER','18.CONTACT_WITH_COVID-19_CASE',
                   '19.ICU','21.SURVIVED']
    
    for col in col_list:
        
        dataframe[col] = le.fit_transform(dataframe[col])
    
    return dataframe


def prepare_split_dataset(df_csv_file,x_column_list,y_column_list,rand_state,train_sz,scaler_name,dataset_name):
    
    df = preprocessing_dataframe(df_csv_file,scaler_name)
    
    df.to_csv(rel_path + '01.Covid19MPD.(Pos_valid_bin_21_scaled['+scaler_name+']_labeled).csv',index = False)
    
    dataset = df.values
    
    X = dataset[:,x_column_list]
    #X = X.astype('int')
    
    y = dataset[:,y_column_list]
    #y = y.astype('int')
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=rand_state,train_size=train_sz)
    
    print('\n'+'================[' + dataset_name + ']================' + '\n' +
          'train size X :',X_train.shape,'\n' + 
          'test size X :',X_test.shape,'\n' +
          'train size y :',y_train.shape,'\n' + 
          'test size y :',y_test.shape,'\n' +
          '==============================================' + '\n')
    
    return X_train,X_test,y_train,y_test,dataset,df



def prepare_data(df_csv_file,x_column_list,y_column_list,rand_state,train_sz,scaler_name,dataset_name):
    
    
    '''
    dataframe = pd.read_csv(df_csv_file)
    
    if  scaler_name == 'std':
        scaler = StandardScaler()
    
    elif scaler_name == 'mm':
        scaler = MinMaxScaler(feature_range=(0,1))
    
    dataframe[['06.AGE']] = scaler.fit_transform(dataframe[['06.AGE']])
    
    dataframe[['20.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION']] = scaler.fit_transform(dataframe[['20.DAYS_FROM_SYMPTOM_TO_HOSPITALIZATION']])
    
    dataframe = dataframe.drop(columns=['01.REGISTRATION_ID','03.OUTPATIENT'])
    
    #dataframe_19 = dataframe_19.sample(frac=1).reset_index(drop=True)
    
    dataframe.to_csv(rel_path + '01.Covid19MPD.(Pos_valid_bin_19_scaled['+scaler_name+']).csv', index = False)
    
    col_list = ['02.SEX','04.INTUBATED','05.PNEUMONIA','08.DIABETIC','09.COPD','10.ASTHMA',
                   '11.IMMUNOSUPPRESSED','12.HYPERTENSION','13.OTHER_CHRONIC_DISEASE','14.CARDIOVASCULAR',
                   '15.OBESITY','16.CHRONIC_KIDNEY_FAILURE','17.SMOKER','18.CONTACT_WITH_COVID-19_CASE',
                   '19.ICU','21.SURVIVED']
    
    label_encoder = LabelEncoder()
    
    onehot_encoder = OneHotEncoder(sparse=False)
    '''
    #pd.get_dummies(dataframe_19, columns=['07.PREGNANCY'])
    #
    # Transform feature gender and degree using one-hot-encoding; Drop the first dummy feature
    #
    #pd.get_dummies(dataframe_19, columns=['07.PREGNANCY'], drop_first=True)
    
    #ct = ColumnTransformer([('one-hot-encoder', OneHotEncoder(), ['07.PREGNANCY'])], remainder='passthrough')
    #
    # For OneHotEncoder with drop='first', the code would look like the following
    #
    #ct2 = ColumnTransformer([('one-hot-encoder', OneHotEncoder(drop='first'), ['07.PREGNANCY'])], remainder='passthrough')
    #
    # Execute Fit_Transform
    #
    #ct.fit_transform(dataframe_19)
    #ct2.fit_transform(dataframe_19)
    # creating initial dataframe
    #bridge_types = (0,1,2)
    #bridge_df = pd.DataFrame(bridge_types, columns=['07.PREGNANCY'])# generate binary values using get_dummies
    #dum_df = pd.get_dummies(bridge_df, columns=["Bridge_Types"], prefix=["Type_is"] )# merge with main df bridge_df on key values
    #bridge_df = bridge_df.join(dum_df)
    #bridge_df
    
    # creating instance of one-hot-encoder
    #enc = OneHotEncoder(handle_unknown='ignore')# passing bridge-types-cat column (label encoded values of bridge_types)
    #enc_df = pd.DataFrame(enc.fit_transform(dataframe_19[['07.PREGNANCY']]).toarray())# merge with main df bridge_df on key values
    #dataframe_19['07.PREGNANCY']
    #dataframe_19 = dataframe_19.join(enc_df)
    
    #dataframe_19
    
    #enc = OneHotEncoder(handle_unknown='ignore')
    #X = [['Male', 1], ['Female', 3], ['Female', 2]]
    #enc.fit(X)
    #dataframe_19['07.PREGNANCY'] = label_encoder.fit_transform(dataframe_19['07.PREGNANCY'])
    '''
    dataframe = pd.get_dummies(dataframe, columns=['07.PREGNANCY'])
    
    
    for col in col_list:
        
        dataframe[col] = label_encoder.fit_transform(dataframe[col])
    
    
    df_pregnancy = pd.DataFrame()
    
    df_pregnancy['07.PREGNANCY_men']= np.where(dataframe['07.PREGNANCY']==97,1,0)
    df_pregnancy['07.PREGNANCY_women_not_pregnant']= np.where(dataframe['07.PREGNANCY']==0,1,0)
    df_pregnancy['07.PREGNANCY_women_pregnant']= np.where(dataframe['07.PREGNANCY']==1,1,0)
    
    dataframe = dataframe.drop(columns=['07.PREGNANCY'])
    dataframe = dataframe.join(df_pregnancy)
    '''
    df = preprocessing_dataframe(df_csv_file,scaler_name)
    
    df.to_csv(rel_path + '01.Covid19MPD.(Pos_valid_bin_21_scaled['+scaler_name+']_labeled).csv',index = False)
    
    dataset = df.values
    
    X = dataset[:,x_column_list]
    #X = X.astype('int')
    
    y = dataset[:,y_column_list]
    #y = y.astype('int')
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=rand_state,train_size=train_sz)
    
    print('\n'+'================[' + dataset_name + ']================' + '\n' +
          'train size X :',X_train.shape,'\n' + 
          'test size X :',X_test.shape,'\n' +
          'train size y :',y_train.shape,'\n' + 
          'test size y :',y_test.shape,'\n' +
          '==============================================' + '\n')
    
    return X_train,X_test,y_train,y_test,dataset,df


#Extract the Optimal Hyperparameter Classification Report's text as String
def create_opt_params_text(method_name,dataset_name,param_grid_dict,cl_report,precision_sc,recall_sc,accur_sc,f1_sc,conf_mtrx,bst_params):
    
    report_text = f'====[Optimum {method_name} Results & Hyperparameters({dataset_name})]====\n\nParameter Grid Used:\n{param_grid_dict}\n\n--------------------[Classification Report]--------------------\n{cl_report}\n---------------------------------------------------------------\nPrecision score:{precision_sc}\nRecall score:{recall_sc}\nAccuracy score:{accur_sc}\nF1 Score:{f1_sc}\n---------------------------------------------------------------\nConfusion Matrix:\n{conf_mtrx}\n---------------------------------------------------------------\nOptimal Parameters:\n{bst_params}\n---------------------------------------------------------------\n'
    
    return report_text


#Write the Optimal Hyperparameter Classification Report's String to a *.txt File(and return the duration{end-start})
def create_write_in_text_file(method_name,report_text,start):
    
    duration_text = ''
    
    file_path = 'txt/' + method_name + '.txt'
    
    with open(file_path, 'a+') as file_object:
        
        file_object.seek(0)
        
        data = file_object.read(100)
        
        if len(data) > 0 :
            
            file_object.write('\n'+'\n')
            
        file_object.write(report_text)
        
        end = time.time()
        
        duration_text = f'Runtime: {end-start} seconds\n===============================================================\n'
        
        file_object.write(duration_text)
    
    return duration_text


#Find the Optimal Hyperparameters for a Model
def find_model_opt_param(X_train,X_test,y_train,y_test,method,param_grid_dict,dataset_name):
    
    start = time.time()
    
    model,method_name = create_model(method,{})

    pipe = Pipeline(steps=[(method, model)], memory='tmp')

    new_param_grid_dict = {}

    for key in param_grid_dict:
        
        curr_new_key_name = ''
        curr_new_key_name = method+'__'+str(key)
        new_param_grid_dict[curr_new_key_name] = param_grid_dict[key]
    
    estimator = GridSearchCV(pipe,param_grid=new_param_grid_dict,cv=10,n_jobs=-1)
    #estimator = GridSearchCV(model,param_grid=param_grid_dict,cv=10,n_jobs=-1)
    
    estimator.fit(X_train, y_train)
    optimum_model_y_pred = estimator.predict(X_test)

    cl_report,precision_sc,recall_sc,accur_sc,f1_sc,conf_mtrx,bst_params = classification_report_opt_metrics(estimator,y_test,optimum_model_y_pred)
    
    report_text = create_opt_params_text(method_name,dataset_name,param_grid_dict,cl_report,precision_sc,recall_sc,accur_sc,f1_sc,conf_mtrx,bst_params)
    
    print(report_text)
    
    duration_text = create_write_in_text_file(method_name,report_text,start)
    
    print(duration_text)


rel_path = 'csv/'





'''Feature Correlation(Pearson-Spearman-Kendall)'''

feature_correlation_std(rel_path + '01.Covid19MPD.(Pos_valid_bin_21).csv','21.SURVIVED')



'''Train & Test sets'''

#Default_features[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
#X_train,X_test,y_train,y_test,dataset,dataframe = prepare_dataset_new(rel_path + '01.Covid19MPD.(Pos_valid_bin_21).csv',
#                                                                     [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,19],
#                                                                     17,42,0.7,'Default_features')

#All_features[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
#X_train_all,X_test_all,y_train_all,y_test_all,dataset_all,dataframe_all = prepare_dataset_new(rel_path + '01.Covid19MPD.(Pos_valid_bin_21).csv',
#                                                                                             [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,19],
#                                                                                             17,42,0.7,'All_features')

X_train,X_test,y_train,y_test,dataset_all,dataframe_all = prepare_split_dataset(rel_path + '01.Covid19MPD.(Pos_valid_bin_21).csv',
                                                                                              [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22],
                                                                                              19,42,0.7,'std','All_features')

X_train_all,X_test_all,y_train_all,y_test_all,dataset_all,dataframe_all = prepare_split_dataset(rel_path + '01.Covid19MPD.(Pos_valid_bin_21).csv',
                                                                                              [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22],
                                                                                              19,42,0.7,'mm','All_features')


#Selected_features_01 [1,3,5,8,15]
#X_train_sel_01,X_test_sel_01,y_train_sel_01,y_test_sel_01,dataset_sel_01,dataframe_sel_01 = prepare_dataset_new(rel_path + '01.Covid19MPD.(Pos_valid_bin_21).csv',
#                                                                                                               [0,2,4],
#                                                                                                               17,42,0.7,'Selected_features_01')


from numpy import hstack
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
 
# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y
 
# get a list of base models
def get_models():
    models = list()
    models.append(('lr', LogisticRegression()))
    models.append(('cart', DecisionTreeClassifier()))
    models.append(('mlp', MLPClassifier()))
    models.append(('randf', RandomForestClassifier()))
    #models.append(('knn', KNeighborsClassifier()))
	#models.append(('svm', SVC(probability=True)))
	#models.append(('bayes', GaussianNB()))
    return models
 
# fit the blending ensemble
def fit_ensemble(models, X_train, X_val, y_train, y_val):
	# fit all models on the training set and predict on hold out set
	meta_X = list()
	for name, model in models:
		# fit in training set
		model.fit(X_train, y_train)
		# predict on hold out set
		yhat = model.predict_proba(X_val)
		# store predictions as input for blending
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# define blending model
	blender = LogisticRegression()
	# fit on predictions from base models
	blender.fit(meta_X, y_val)
	return blender
 
# make a prediction with the blending ensemble
def predict_ensemble(models, blender, X_test):
	# make predictions with base models
	meta_X = list()
	for name, model in models:
		# predict with base model
		yhat = model.predict_proba(X_test)
		# store prediction
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# predict
	return blender.predict(meta_X)



   
# define dataset
dataframe = pd.read_csv(rel_path + '01.Covid19MPD.(Pos_valid_bin_21).csv')
dataset = dataframe.values
    
X = dataset[:,[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
X = X.astype('int')
    
y = dataset[:,20]
y = y.astype('int')

# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# split training set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=42)
# summarize data split
print('Train: %s, Val: %s, Test: %s' % (X_train.shape, X_val.shape, X_test.shape))
# create the base models
models = get_models()
# train the blending ensemble
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
# make predictions on test set
yhat = predict_ensemble(models, blender, X_test)
# evaluate predictions
score = accuracy_score(y_test, yhat)
print('Blending Accuracy: %.3f' % (score*100))



from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

def evaluate(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    print("TRAINIG RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

    print("TESTING RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")



'''
from brew.base import Ensemble
from brew.base import EnsembleClassifier
from brew.combination.combiner import Combiner


clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf3 = RandomForestClassifier()
clf4 = MLPClassifier()
clf5 = KNeighborsClassifier()

# create your Ensemble
clfs = [clf1,clf2,clf3,clf4]
ens = Ensemble(classifiers = clfs)

# create your Combiner
# the rules can be 'majority_vote', 'max', 'min', 'mean' or 'median'
comb = Combiner(rule='mean')

# now create your ensemble classifier
ensemble_clf = EnsembleClassifier(ensemble=ens, combiner=comb)
ensemble_clf.predict(X_test)
'''


ada_boost_clf = AdaBoostClassifier(n_estimators=30)
ada_boost_clf.fit(X_train, y_train)
evaluate(ada_boost_clf, X_train, X_test, y_train, y_test)



grad_boost_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
grad_boost_clf.fit(X_train, y_train)
evaluate(grad_boost_clf, X_train, X_test, y_train, y_test)



estimators = []
log_reg = LogisticRegression()
estimators.append(('Logistic',log_reg))

d_tree = DecisionTreeClassifier()
estimators.append(('Tree',d_tree))

rand_f = RandomForestClassifier()
estimators.append(('RandomF',rand_f))

mlp = MLPClassifier()
estimators.append(('MLP',mlp))

#knbrs = KNeighborsClassifier()
#estimators.append(('KNeighbors',knbrs))


voting = VotingClassifier(estimators=estimators)
voting.fit(X_train, y_train)

evaluate(voting, X_train, X_test, y_train, y_test)


scores = {'Train': accuracy_score(y_train, voting.predict(X_train)),'Test': accuracy_score(y_test, voting.predict(X_test)),}


'''Random Forest'''


'''All Features'''

'''Default_All'''
model_create_train_pred_analysis(X_train_all,X_test_all,y_train_all,y_test_all,'rf',
                                 {'bootstrap':True,'ccp_alpha':0.0,'class_weight':None,'criterion':'gini',
                                  'max_depth':20, 'max_features':'auto','max_leaf_nodes':None,'max_samples':None,
                                  'min_impurity_decrease':0.0,'min_samples_leaf':5,'min_samples_split':5,
                                  'min_weight_fraction_leaf':0.0,'n_estimators':100,
                                  'oob_score':False,'random_state':None,'verbose':0,'warm_start':False},
                                 'Default_All')





'''Selected_features_01 [1,3,5,8,15]'''

'''
#Default_Selected_features
model_create_train_pred_analysis(X_train_sel_01,X_test_sel_01,y_train_sel_01,y_test_sel_01,'rf',
                                 {'bootstrap':True,'ccp_alpha':0.0,'class_weight':None,'criterion':'gini',
                                  'max_depth':20, 'max_features':'auto','max_leaf_nodes':None,'max_samples':None,
                                  'min_impurity_decrease':0.0,'min_samples_leaf':5,'min_samples_split':5,
                                  'min_weight_fraction_leaf':0.0,'n_estimators':100,
                                  'oob_score':False,'random_state':None,'verbose':0,'warm_start':False},
                                 'Default_Selected_features')

'''

'''#Find Optimal Hyperparameters'''
'''
#Tester
find_model_opt_param(X_train,X_test,y_train,y_test,'rf',
                     {'bootstrap':[True],'ccp_alpha':[0.0],
                      'class_weight':[None],'criterion':['gini','entropy'],
                      'max_depth':[None],'max_features':['auto','sqrt'],
                      'max_leaf_nodes':[None],'max_samples':[None],'min_impurity_decrease':[0.0],
                      'min_samples_leaf':[1],'min_samples_split':[2],
                      'min_weight_fraction_leaf':[0.0],'n_estimators':[100],
                      'oob_score':[False],'random_state':[42],'verbose':[0],'warm_start':[False]},
                     'Tester')

'''