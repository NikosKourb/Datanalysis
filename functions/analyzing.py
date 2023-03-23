import pandas as pd
#import numpy as np

from classes.Column import Column

from functions.plot.graph_plotting import df_graph_plot_show_save

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



'''Analyzing Dataset Functions'''

#Show the number of NaN and Non Unique values of a column
def show_column_info(df,column_name):
    
    row_number = len(df.index)
    null_count = df[column_name].isnull().sum()
    unique_count = df[column_name].nunique()
    non_unique_count = row_number - (null_count + unique_count)
    data_type = df[column_name].dtypes
        
    print('\n' + '-----[' + df.name + ']-----' +
          '\n' + 'Column ' + column_name + ' [' + str(data_type) + ']' +
          '\n' + 'has ' + str(row_number) + ' rows, containing:' +
          '\n' + str(null_count) + ' NaN Values' +
          '\n' + str(unique_count) + ' Unique Values' +
          '\n' + str(non_unique_count) + ' Non Unique Values' +
          '\n')


#Show the number of NaN and Non Unique values in all the columns of a dataframe
def show_all_column_info(df):
    
    column_name_list = list(df.columns.values)

    for column_name in column_name_list:
    
        show_column_info(df,column_name)


#Analysis of a df column with categorical values
def categorical_column_analysis(df,column):
    
    result_list = []
    data_type = df[column.name].dtypes
    row_number = len(df.index)
    null_count = df[column.name].isnull().sum()
    
    for char_key in column.characteristics_dict:
        
        char_name = char_key
        char_value=column.characteristics_dict[char_key]
        char_quantity = (df[column.name]== char_value).sum()
        result_list.append([char_name,char_value,char_quantity])
        
    print('\n' + '-----[' + df.name + ']-----' +
          '\n' + 'Column ' + column.name + ' [' + str(data_type) + ']' +
          '\n' + 'has ' + str(row_number) + ' rows, containing:')

    for j in range(len(result_list)):

        var_name_r = result_list[j][0]
        var_value_r =result_list[j][1]
        var_name_count_r =result_list[j][2]

        print(str(var_name_count_r) + ' '+ str(var_name_r) + '(' + str(var_value_r) + ')')

    print(str(null_count) + ' NaN values' +
          '\n' + '--------------------' + '\n')


#Create a barchart from a dataframe column with categorical values (df,column_name,var_value_list,var_name_list):
def create_categorical_column_bar_chart(df,column,graph_title):
    
    my_dictionary={}
    index = [column.my_name]
    
    for key in column.characteristics_dict:

        #value_quantity = int(df.loc[df[column.name]==column.characteristics_dict[key], :][column.name].count())
        
        df_filter = df[df[column.name]==column.characteristics_dict[key]]

        filter_index = df_filter.index
        value_quantity = len(filter_index)
        
        my_dictionary[key] = value_quantity
    
    my_dictionary['NaN'] = df[column.name].isnull().sum()
    df = pd.DataFrame(my_dictionary, index=index)
    
    graph_title_ = graph_title.replace(' ','_')
    
    df_graph_plot_show_save(df,graph_title,False,200,'files/png/df_analysis/' + graph_title_ + '.png')
    df_graph_plot_show_save(df,graph_title,True,200,'files/png/df_analysis/' + graph_title_ + '_stacked.png')


#Create a barchart from a dataframe list of columns with categorical values (df,column_name,var_value_list,var_name_list):
def create_multi_df_categorical_column_bar_chart(df,column_list,col_val_list,graph_title):
    
    my_dictionary = dict.fromkeys(col_val_list)
    
    for key in my_dictionary:
        curr_list=[]
        my_dictionary[key]=curr_list

    index = []

    for col in column_list:

        index.append(col.my_name)
        
        for key in col.characteristics_dict:
            
            #value_quantity = int(df.loc[df[col.name]==col.characteristics_dict[key], :][col.name].count())
            
            df_filter = df[df[col.name]==col.characteristics_dict[key]]

            filter_index = df_filter.index
            value_quantity = len(filter_index)
            
            my_dictionary[key].append(value_quantity)
            
    df = pd.DataFrame(my_dictionary, index=index)
    
    graph_title_ = graph_title.replace(' ','_')
    
    df_graph_plot_show_save(df,graph_title,False,200,'files/png/df_analysis/' + graph_title_ + '.png')
    df_graph_plot_show_save(df,graph_title,True,200,'files/png/df_analysis/' + graph_title_ + '_stacked.png')


#Create a barchart from a dataframe list of columns with categorical values as an addage to the above function
def create_multi_categorical_column_list_chart(df,col_dict,graph_title):
    
    my_column_values_list=['yes','no','not apply','ignored','not specified']
    dict_common ={'yes':1,'no':2,'not apply':97,'ignored':98,'not specified':99}
    my_col_list=[]
    
    for col in col_dict:
        my_col_list.append(Column(str(col),str(col_dict[col]),dict_common))
    
    create_multi_df_categorical_column_bar_chart(df,my_col_list,my_column_values_list,graph_title)


#Create a barchart from a dataframe list of columns with numerical & categorical values (df,column_list_a,col_val_list_a,column_list_b,col_val_list_b):
def create_multi_df_numerical_column_bar_chart(df,column_list_a,col_val_list_a,column_list_b,col_val_list_b,graph_title):
    
    my_dict = dict.fromkeys(col_val_list_b)
    
    for key in my_dict:
        curr_list=[]
        my_dict[key]=curr_list

    index = []
    
    for col_a in column_list_a:

        index.append(col_a.my_name)
        
        for key_a in col_a.characteristics_dict:
            
            for col_b in column_list_b:
                
                for key_b in col_b.characteristics_dict:
                    
                    df_filter = df[(df[col_a.name] >= col_a.characteristics_dict[key_a][0]) & 
                                            (df[col_a.name] <= col_a.characteristics_dict[key_a][1]) &
                                            (df[col_b.name]==col_b.characteristics_dict[key_b])]

                    filter_index = df_filter.index
                    value_quantity = len(filter_index)

                    my_dict[key_b].append(value_quantity)
            
    df = pd.DataFrame(my_dict, index=index)

    graph_title_ = graph_title.replace(' ','_')
    
    df_graph_plot_show_save(df,graph_title,False,200,'files/png/df_analysis/' + graph_title_ + '.png')
    df_graph_plot_show_save(df,graph_title,True,200,'files/png/df_analysis/' + graph_title_ + '_stacked.png')
    

#Numerical Column Analysis
def numerical_column_analysis(df,col_name):
    print('\n' ,df[[col_name]].describe(),'\n' + '\n',
          ((df[col_name])< 0).sum(),'Invalid values' + '\n',
          df[col_name].isnull().sum(),'NaN values' + '\n')


#Analyze the metrics from various dfs
def analyze_report_metrics_df(report_all_df_path):
    
    report_all_df = pd.read_csv(report_all_df_path,header=0)
    
    report_param_list = ['Precision','Recall','Accuracy','F1']
    
    for param in report_param_list:
        
        numerical_column_analysis(report_all_df,param)


#Create Ages & Sexes Barchart
def ages_column_barchart(df,col_ages_name,col_sexes_name,graph_title):
    
    column_sexes_values_list= ['women','men','not apply','ignored','not specified']
    column_ages_values_list= ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99','100 and over']
    
    sexes_column_list =[Column(col_sexes_name,'Sexes',{'women':1,'men':2,'not apply':97,'ignored':98,'not specified':99})]
    ages_column_list =[Column(col_ages_name,'0-9',{'0-9':[0,9]}),Column(col_ages_name,'10-19',{'10-19':[10,19]}),
                       Column(col_ages_name,'20-29',{'20-29':[20,29]}),Column(col_ages_name,'30-39',{'30-39':[30,39]}),
                       Column(col_ages_name,'40-49',{'40-49':[40,49]}),Column(col_ages_name,'50-59',{'50-59':[50,59]}),
                       Column(col_ages_name,'60-69',{'60-69':[60,69]}),Column(col_ages_name,'70-79',{'70-79':[70,79]}),
                       Column(col_ages_name,'80-89',{'80-89':[80,89]}),Column(col_ages_name,'90-99',{'90-99':[90,99]}),
                       Column(col_ages_name,'100 and over',{'100 and over':[100,500]})]
    
    create_multi_df_numerical_column_bar_chart(df,ages_column_list,column_ages_values_list,sexes_column_list,column_sexes_values_list,graph_title)


#Ages Column Analysis & Create Ages & Sexes Barchart
def ages_column_analysis_and_barchart(df1,col_ages_name1,col_sexes_name1,graph_title1,df2,col_ages_name2,col_sexes_name2,graph_title2):
    
    numerical_column_analysis(df1,col_ages_name1)
    ages_column_barchart(df1,col_ages_name1,col_sexes_name1,graph_title1)
    
    numerical_column_analysis(df2,col_ages_name2)
    ages_column_barchart(df2,col_ages_name2,col_sexes_name2,graph_title2)
    

#Analyze a categorical values of a df column and plot the corresponding barchart 
def categorical_column_analysis_and_bar_chart(df,col,graph_title):
    
    categorical_column_analysis(df,col)
    create_categorical_column_bar_chart(df,col,graph_title)


#Analyze the categorical values of a certain column and create the corresponding barchart for 2 dfs
def df_compare_categorical_column_analysis_and_bar_chart(col,df1,chart1,df2,chart2):
    
    categorical_column_analysis_and_bar_chart(df1,col,chart1)
    categorical_column_analysis_and_bar_chart(df2,col,chart2)


''''''
