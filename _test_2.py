
import pandas as pd

from functions.models.create_train_predict_analyze import get_model_name
from functions.plot.graph_plotting import series_graph_plot_show_save

from functions.models.analyze_metrics import multi_dataset_model_metrics_total_to_csv
from functions.models.analyze_metrics import multi_df_sort_values


'''Feature Importance Diagrams for MLPs & KNN'''

rel_path = 'files/csv/f_importance/'
df_original = pd.read_csv(rel_path + 'MLPs.and.KNN.csv',header=0)


feat_importances_all = df_original.squeeze()


feat_importances_all = feat_importances_all.reset_index(drop=True)

#new_feat_importances_all = feat_importances_all.copy()

feat_importances_all.set_index('Feature_Name',drop=True,inplace=True)

print(feat_importances_all)

model_dict={'mlp':'Multi_Layer_Perceptrons','knc':'KNeighbors_Classifier'}

for model in model_dict:
    method = get_model_name(model)
    graph_title = method + ' All Features\' Importance Overall' 
    graph_file_path = 'files/png/f_importance/' + method + '/' + model + '_all_feats_total.png'
    series_graph_plot_show_save(feat_importances_all,graph_title,'barh',200,graph_file_path)



'''Analyze Total Model Metrics'''

#Total Metrics according to: Preprocessing(12), Feature Number(3[22,15,12]) & Hypeparameters (3[default,opt-01,opt-02])
multi_dataset_model_metrics_total_to_csv(['lgr','dtc','rfc','xbc','mlp','knc'],#'lgr','dtc','rfc','xbc','mlp','knc','svc'
                                         'prep_features_params',
                                         ['fc_none','fc_std',
                                          'fc_mm_0-1','fc_mm_0-10','fc_mm_0-100','fc_mm_0-1000',
                                          'lr_none','lr_std',
                                          'lr_mm_0-1','lr_mm_0-10','lr_mm_0-100','lr_mm_0-1000'],
                                         ['22_features','15_features','10_features'],
                                         ['default_params','opt-01_params','opt-02_params'])

multi_df_sort_values('lgr',['fc','lr'],'prep_features_params',
                     ['Precision_mean','Recall_mean','Accuracy_mean','F1_mean','ROC_AUC_mean','P_R_AUC_mean','Runtime(seconds)_mean'])
