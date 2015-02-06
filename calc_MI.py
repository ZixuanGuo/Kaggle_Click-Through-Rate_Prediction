# -*- coding: utf-8 -*-
"""
Created on Wed Feb 04 11:35:20 2015

@author: chenj
"""
# library
from sklearn.metrics import mutual_info_score
from operator import itemgetter
import pandas as pd

##########################################
##  directories, files, and parameters  ##
##########################################

# working and data directories
data_dir  = 'C:\\Users\\chenj\\Documents\\analytics\\CTR\\'
chunk_dir = data_dir + 'train_chunks\\'
work_dir = data_dir + 'SGD\\'

# input files
train_file = data_dir + 'train.csv'
test_file  = data_dir + 'test.csv'
test_procd = data_dir + 'test1.csv'
sub_file   = work_dir + 'submission.csv'
sub_zip    = work_dir + 'submission.zip'

# data params
n_test     = 4577464   # num of test data points
M          = 2**20     # parameter for hashing

##########################################
##  function definitions                ##
##########################################
def CalcMI(filename):
    '''FUNCTION CalcMI calculates the MI between features and the classification
    
    INPUT:
        filename: is the file containing the data
    
    OUTPUT:
        a dictionary containing the features and their mutual information with
        the Click
    '''    
    data = pd.read_csv(filename, dtype = object)
    data['hour1'] = data['hour'].apply(lambda x: x[6:])
    data.drop('hour', axis = 1, inplace = True)
    MI_score = dict()
    
    colnames = data.columns
    colnames = colnames[2:] # get rid of id and click
    # calcualte MI for original features
    for col in colnames:
        MI_score[col] = mutual_info_score(data['click'], data[col])
    
    # calculate MI for feature interactions
    for i in range(len(colnames)):
        for j in range(i+1, len(colnames)):
            col_i = colnames[i]
            col_j = colnames[j]
            tmp = data[[col_i, col_j]].apply(lambda x: x[0] + '_' + x[1], axis = 1)
            MI_score[col_i + '*' + col_j] = mutual_info_score(data['click'], tmp)
            print '%s: %0.4f' % (col_i + '*' + col_j, MI_score[col_i + '*' + col_j])
            
    return MI_score        


################################################
##  find out important feature interations    ##
################################################

# calculate mutual information of the original features with click 
# and click with the feature interactions
filename = chunk_dir + 'chunk30.csv'
MI_score = CalcMI(filename)
sorted_score = sorted(MI_score.items(), key=itemgetter(1), reverse = True)

print('=================================================')
for key, val in sorted_score:
    print '%s: %0.5f' % (key, val)
    

with open('inter_score.txt', 'w') as out:
    for key, val in sorted_score:
        out.write('%s:%s\n', key, str(val))


    
