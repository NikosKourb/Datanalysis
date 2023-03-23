#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as mtp
from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import figure

#from classes.Column import Column

#from sklearn.inspection import permutation_importance

#from functions.models.create_train_predict_analyze import model_name
#from functions.models.create_train_predict_analyze import create_model

from sklearn.metrics import roc_curve
#from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_confusion_matrix
#import time

import warnings
warnings.filterwarnings("ignore")

'''Anaconda commands in conda promt''''''
   
conda create -n py37 python=3.7  #for version 3.7
conda activate py37
conda install pip
conda install wheel
conda install pandas
conda install -c conda-forge imbalanced-learn
conda install matplotlib
pip install matplotlib

Confusion Matrix Visualization:
conda install seaborn
pip install seaborn

Visualizing Decision Trees:
pip install graphviz
pip install pydotplus

Installing XGBoost Regression:
pip install xgboost

'''



'''Graph Plotting Functions'''

#Plots Graph, Shows & Saves file from a Series
def series_graph_plot_show_save(series_file,graph_title,graph_kind,graph_file_dpi,graph_file_path):
    
    graph = series_file.plot(kind=graph_kind,title=graph_title)
    
    graph_show_save(graph,graph_file_path,graph_file_dpi)
    


#Plots Graph, Shows & Saves file from a DF
def df_graph_plot_show_save(df_file,graph_title,graph_type_bool,graph_file_dpi,graph_file_path):
    
    graph = df_file.plot.bar(title=graph_title,stacked=graph_type_bool)
    
    graph_show_save(graph,graph_file_path,graph_file_dpi)


#Confusion Matrix Graph
def conf_matrix_plot(model,X_test,y_test,graph_title,graph_file_path,graph_file_dpi):
    cm_graph = plot_confusion_matrix(model, X_test, y_test,
                                     display_labels=['Survived', 'Died'],
                                     cmap='RdPu',#other cmaps: 'GnBu','autumn','Wistia','PuOr','gist_rainbow'
                                     values_format = '.5g')
    
    cm_graph.ax_.set_title('Confusion Matrix' + graph_title)
    cm_graph.ax_.set_xlabel('Predicted Outcome')
    cm_graph.ax_.set_ylabel('Actual Outcome')
    
    cm_graph.figure_.savefig(graph_file_path,bbox_inches='tight',dpi=graph_file_dpi)
    
    plt.show()
 

#ROC Curve Graph
def roc_curve_plot(y_test,no_skill_probs,model_probs,model_name,graph_title,graph_file_path,graph_file_dpi):
    
    ns_fpr, ns_tpr, _ = roc_curve(y_test, no_skill_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, model_probs)
    
    graph = plt.gca()
    
    graph.set_title(graph_title)
    
    graph.plot(ns_fpr, ns_tpr, linestyle='--',label='No Skill')
    graph.plot(lr_fpr, lr_tpr, marker='.',label=model_name)
    
    graph.set_xlabel('False Positive Rate')
    graph.set_ylabel('True Positive Rate')
    
    graph_show_save(graph,graph_file_path,graph_file_dpi)


#Precision-Recall Curve Graph
def precision_recall_curve_plot(y_test,model_precision,model_recall,model_name,graph_title,graph_file_path,graph_file_dpi):
    
    graph = plt.gca()
    
    graph.set_title(graph_title)
    
    no_skill = len(y_test[y_test==1]) / len(y_test)
    
    graph.plot([0, 1], [no_skill, no_skill], linestyle='--',label='No Skill')
    graph.plot(model_recall, model_precision, marker='.',label=model_name)

    graph.set_xlabel('Recall')
    graph.set_ylabel('Precision')
    
    graph_show_save(graph,graph_file_path,graph_file_dpi)


#Shows & Saves Graph
def graph_show_save(graph,graph_file_path,graph_file_dpi):
    
    graph.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    
    fig = graph.get_figure()
    
    fig.savefig(graph_file_path,bbox_inches='tight',dpi = graph_file_dpi)
    
    plt.show()


''''''
