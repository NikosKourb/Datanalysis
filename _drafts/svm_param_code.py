from Column import Column#[created class]
#from my_functions import *
from my_functions import feature_correlation_std
from my_functions import prepare_dataset
from my_functions import model_create_train_pred_analysis
from my_functions import find_model_opt_param
from sklearn.svm import SVC



'''

covid19_df = pd.read_csv('01.Covid19MPD.(Pos_valid_bin_21).csv')
print(covid19_df.head())
print(covid19_df.info())
print(covid19_df.describe())

scaler = StandardScaler()
ros = RandomOverSampler()
scaler = MinMaxScaler()

Pipeline:
steps = [('scaler', scaler),
         ('sample', ros),
         ('SVM', svm_clf)]
pipe = Pipeline(steps=steps, memory='tmp')

'SVM__gamma': gamma

estimator = GridSearchCV(pipe, param_grid=params, cv=10, scoring='precision', n_jobs=-1)

'''



rel_path = 'csv/'


'''Feature Correlation(Pearson-Spearman-Kendall)'''

feature_correlation_std(rel_path + '01.Covid19MPD.(Pos_valid_bin_21).csv','21.SURVIVED')



'''Train & Test sets'''

#Default_features[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
X_train,X_test,y_train,y_test = prepare_dataset(rel_path + '01.Covid19MPD.(Pos_valid_bin_21).csv',
                                                [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                                                20,42,0.7,'Default_features')

#All_features[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
X_train_all,X_test_all,y_train_all,y_test_all = prepare_dataset(rel_path + '01.Covid19MPD.(Pos_valid_bin_21).csv',
                                                                [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                                                                20,42,0.7,'All_features')

#Selected_features_01 [1,3,5,8,15]
X_train_sel_01,X_test_sel_01,y_train_sel_01,y_test_sel_01 = prepare_dataset(rel_path + '01.Covid19MPD.(Pos_valid_bin_21).csv',
                                                                            [1,3,5,8,15],
                                                                            20,42,0.7,'Selected_features_01')

model= SVC()
print(model.get_params())


'''Support Vector Machines'''


'''All Features'''

'''Default_All'''
model_create_train_pred_analysis(X_train_all,X_test_all,y_train_all,y_test_all,'svm',
                                 {'C':1.0,'break_ties':False,'cache_size':200,'class_weight':None,
                                  'coef0':0.0,'decision_function_shape':'ovr','degree':3,'gamma':'scale',
                                  'kernel':'rbf','max_iter':-1,'probability':False,'random_state':None,
                                  'shrinking':True,'tol':0.001,'verbose':False},
                                 'Default_All')


'''Optimal_All_01'''
model_create_train_pred_analysis(X_train_all,X_test_all,y_train_all,y_test_all,'svm',
                                 {},
                                 'Optimal_All_01')

'''Optimal_All_02'''
model_create_train_pred_analysis(X_train_all,X_test_all,y_train_all,y_test_all,'svm',
                                 {},
                                 'Optimal_All_02')


'''Selected_features_01 [1,3,5,8,15]'''

'''Default_Selected_features'''
model_create_train_pred_analysis(X_train_sel_01,X_test_sel_01,y_train_sel_01,y_test_sel_01,'svm',
                                 {'C':1.0,'break_ties':False,'cache_size':200,'class_weight':None,
                                  'coef0':0.0,'decision_function_shape':'ovr','degree':3,'gamma':'scale',
                                  'kernel':'rbf','max_iter':-1,'probability':False,'random_state':None,
                                  'shrinking':True,'tol':0.001,'verbose':False},
                                 'Default_Selected_features')

'''Optimal_Selected_features_01'''
model_create_train_pred_analysis(X_train_sel_01,X_test_sel_01,y_train_sel_01,y_test_sel_01,'svm',
                                 {},
                                 'Optimal_Selected_features_01')


'''Optimal_Selected_features_02'''
model_create_train_pred_analysis(X_train_sel_01,X_test_sel_01,y_train_sel_01,y_test_sel_01,'svm',
                                 {},
                                 'Optimal_Selected_features_02')


'''Find Optimal Hyperparameters'''

#Tester
find_model_opt_param(X_train,X_test,y_train,y_test,'svm',
                     {'C':[0.001,1.0],'break_ties':[False],'cache_size':[200],'class_weight':[None],
                      'coef0':[0.0],'decision_function_shape':['ovr','ovo'],'degree':[3],'gamma':['scale','auto'],
                      'kernel':['rbf'],'max_iter':[-1],'probability':[False],'random_state':[None],
                      'shrinking':[True],'tol':[0.001],'verbose':[False]},
                     'Tester')

#Default_features
find_model_opt_param(X_train,X_test,y_train,y_test,'svm',
                     {},
                     'Default')

#All_features
find_model_opt_param(X_train_all,X_test_all,y_train_all,y_test_all,'svm',
                     {},
                     'All_features')

#Selected_features_01 [1,3,5,8,15]
find_model_opt_param(X_train_sel_01,X_test_sel_01,y_train_sel_01,y_test_sel_01,'svm',
                     {},
                     'Selected_features_01')

''''''
