from Column import Column#[created class]
#from my_functions import *
from my_functions import feature_correlation_std
from my_functions import prepare_dataset
from my_functions import model_create_train_pred_analysis
from my_functions import find_model_opt_param



'''

covid19_df = pd.read_csv('01.Covid19MPD.(Pos_valid_bin_21).csv')
print(covid19_df.head())
print(covid19_df.info())
print(covid19_df.describe())

scaler = StandardScaler()
ros = RandomOverSampler()
scaler = MinMaxScaler()

'''



'''Feature Correlation(Pearson-Spearman-Kendall)'''

feature_correlation_std('01.Covid19MPD.(Pos_valid_bin_21).csv','21.SURVIVED')



'''Train & Test sets'''

#Default_features[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
X_train,X_test,y_train,y_test = prepare_dataset('01.Covid19MPD.(Pos_valid_bin_21).csv',
                                                [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                                                20,42,0.7,'Default_features')

#All_features[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
X_train_all,X_test_all,y_train_all,y_test_all = prepare_dataset('01.Covid19MPD.(Pos_valid_bin_21).csv',
                                                                [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                                                                20,42,0.7,'All_features')

#Selected_features_01 [1,3,5,8,15]
X_train_sel_01,X_test_sel_01,y_train_sel_01,y_test_sel_01 = prepare_dataset('01.Covid19MPD.(Pos_valid_bin_21).csv',
                                                                            [1,3,5,8,15],
                                                                            20,42,0.7,'Selected_features_01')


