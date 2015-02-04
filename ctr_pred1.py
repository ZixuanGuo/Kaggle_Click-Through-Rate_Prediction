"""
@author: Jieqiu Chen, jieqiu0808@gmail.com

@note:  the purpose of this script is to build a prediction model for the 
        folowing kaggle competition:
            http://www.kaggle.com/c/avazu-ctr-prediction, 
        which predicts whether a mobile ad will be clicked.
        
        Methodology:
            - Here we use hash trick to transform the feature and random forest
            to train the model. 
            - The final prediction is an ensemble of 50 predictions generated
            by training random forest model on 50 dataset. Each of the 50 data
            set is a subset of the orignal training data.
"""
import pandas as pd
import random as rd
import os
import csv
import zipfile
from math import log
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

##########################################
##  directories, files, and parameters  ##
##########################################

# working and data directories
work_dir  = 'C:\\Users\\chenj\\Documents\\analytics\\CTR\\'
data_dir  = work_dir + 'train_chunks\\'
procd_dir = work_dir + 'processed\\' 
tmp_dir   = work_dir + 'pred\\'

# input files
train_file = work_dir + 'train.csv'
test_file  = work_dir + 'test.csv'
test_procd = work_dir + 'test1.csv'
pred_file  = work_dir + 'prediction.csv'
sub_file   = work_dir + 'submission_0.csv'
sub_zip    = work_dir + 'submission.zip'


# control params
chunk_data   = False     # divide original train data into small chunks
process_data = False     # turn raw data into data ready to train
fast_test    = False     # quickly test the code, using only one chunk of data to build model and make prediction
full_test    = True      # perform CV and prediction on test data
submit       = True      # ensemble prediction, and make submission file

# data params
n_test_cases = 4577464   # num of test data points
CHUNK_SIZE   = 10000     # num of rows to read using pd.read_csv
CHUNK_NUM    = 50        # num of chunks train data to divide into 
M            = 2**20     # parameter for hashing

# model params
kfold    = 5             # num of folds in cross validation
ntree    = 20            # num of trees in random forest classifier

colnames = ['id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category',
            'app_id', 'app_domain', 'app_category', 
            'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 
            'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']    
            
features = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category',
            'app_id', 'app_domain', 'app_category', 
            'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 
            'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
            
##########################################
##  function definitions                ##
##########################################
def divide_data(train_file, CHUNK_NUM = CHUNK_NUM):
    ''' FUNCTION: dvide_data performs random shuffle of the train.csv 
                  and divides the original data into smaller chunks
        
        INPUT:
            train_file: file path for train.csv
            CHUNK_NUM: num of chunks to divide
        
        OUTPUT:
            CSV files each is small chunk of original train.csv
    '''
    fnames = ['chunk' + str(i) + '.csv' for i in range(CHUNK_NUM)]

    tp = pd.read_csv(train_file, iterator = True, chunksize = CHUNK_SIZE)
    
    for i, df in enumerate(tp):
        idx = range(len(df))
        rd.shuffle(idx)
        idx = [ i % CHUNK_NUM for i in idx]
             
        if i == 0:
            for j in range(CHUNK_NUM):
                this_idx = [i == j for i in idx]
                tmp = df[this_idx]
                tmp.to_csv(data_dir + fnames[j], index = False)
        else:
            for j in range(CHUNK_NUM):
                this_idx = [i == j for i in idx]
                tmp = df[this_idx]
                tmp.to_csv(data_dir + fnames[j], index = False, mode = 'a', header = False)


def process_raw(csv_in, csv_out, test = False):
    ''' FUNCTION: one hot encode the categorical features
        
        INPUT:
            csv_in: input raw data file path
            csv_out: output file path
        
        OUTPUT: none
    '''
    with open(csv_in, 'rb') as infile, open(csv_out, 'wb') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, colnames, extrasaction='ignore')
        writer.writeheader()

        for row in reader:
            if test == False:
                if row['click'] == '0':
                    row['click'] = 0
                else:
                    row['click'] = 1
                
            row['hour'] = row['hour'][6:]
            cols = features[1:]

            for col in cols:
                value = row[col]
                row[col] = abs(hash(col + value)) % M

            writer.writerow(row)


def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction, an array
            y: real answer, an array

        OUTPUT:
            logarithmic loss of p given y
    '''
    p = [max(min(pp, 1. - 10e-15), 10e-15) for pp in p]
    loss = [-log(1. - pp) for pp in p]

    for i in xrange(len(y)):
        if y[i] == 1:
            loss[i] = -log(p[i])

    return np.mean(loss)

 
def CV_test(fname):
    ''' FUNCTION: cross validation and perform prediction

        INPUT:
            fname: filename to be used for training

        OUTPUT:
            cross validated avg logloss score
    '''
    data = pd.read_csv(procd_dir + fname)
    rf = RandomForestClassifier(n_estimators=ntree)
    
    fname.replace('procd', 'pred')
    outfname = tmp_dir + fname
    
    # cv = cross_validation.KFold(len(data), n_folds=kfold, indices = False)
    cv = cross_validation.KFold(len(data), n_folds=kfold)
    results = []
    for trainidx, testidx in cv:
        probs = rf.fit(data.ix[trainidx, features], data.ix[trainidx, 'click']).predict_proba(data.ix[testidx, features])
        results.append(logloss(data.ix[testidx, 'click'], [x[1] for x in probs]))    
    
    rf.fit(data[features], data['click'])    
    with open(outfname, 'w') as outfile, open(test_procd, 'r') as infile:
        test_reader = pd.read_csv(infile, iterator = True, chunksize = CHUNK_SIZE)        
    
        for i, df in enumerate(test_reader):
            probs = rf.predict_proba(df[features])
            df['click'] = [x[1] for x in probs]
            if i > 0:
                df.to_csv(outfile, cols = ['id', 'click'], mode = 'a', header = False, index = False)
            else:
                df.to_csv(outfile, cols = ['id', 'click'], index = False)
                
    return np.mean(results)                         
                     
##############################################
##  random shuffle and pre-process          ## 
##############################################

# whether to divide the original data into smaller chunks
if chunk_data == True:
    
    print('Shuffling data and divide them into smaller chunks ...')
    
    if os.path.isdir(data_dir) == False:
        os.mkdir(data_dir)
        
    divide_data(train_file)

# whether to pre-process the data
if process_data == True:
    
    print('Pre-processing data ...')

    if os.path.isdir(procd_dir) == False:
        os.mkdir(procd_dir)
    
    fnames = os.listdir(data_dir)

    for fname in fnames:
        csv_in  = data_dir + fname
        csv_out = procd_dir + fname.replace('chunk','procd')
        process_raw(csv_in, csv_out)
    
    # process test data
    process_raw(work_dir+'test.csv', work_dir+'test1.csv', True)
                     
##############################################
##  train through random forest classifier  ## 
##############################################
if fast_test:
    fnames = os.listdir(procd_dir)
    fname = fnames[0]    
    data = pd.read_csv(procd_dir + fname)
    y = data['click']
    X = data[features]
    
    # create Stratified k-fold
    cv = cross_validation.StratifiedKFold(y, kfold)
    
    # create random foreast model
    rf = RandomForestClassifier(n_estimators=ntree)
    log_loss = []
    
    for train_idx, test_idx in cv:
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = rf.predict_proba(X.iloc[test_idx])  
        prob = [p[1] for p in prob]
        loss = logloss(prob, y[test_idx].values)
        log_loss.append(loss)
        print("in this fold, logloss = %.2f", loss)
    
    print("average logloss = %.2f", np.mean(log_loss))
    

##############################################
##   make prediction on test data           ##
##############################################
if not os.path.isfile(work_dir + 'test1.csv'):
    # process raw test data and make it the same format as training
    process_raw(work_dir + 'test.csv', work_dir + 'test1.csv', True)

if full_test == True:    
    fnames = os.listdir(procd_dir)
    cv_results = []

    for fname in fnames:
        print('================================')
        print('Training on data: ' + fname + ' ...')
        print('================================')
        res = CV_test(fname)
        cv_results.append(res)
        print('This round avg logloss: %0.2f' % res)  
    
    cv_results = np.array(cv_results)
    print 'Mean and Std of Cross Validation: %0.2f, %0.2f' % (cv_results.mean(), cv_results.std()) 
    
    
##############################################
## combine results to form submission file  ##
##############################################

if submit == True:
    fnames = [tmp_dir + 'procd' + str(i) + '.csv' for i in range(CHUNK_NUM)]
    rounds = n_test_cases/CHUNK_SIZE + 1
    skip = 1
    header = ['id', 'click']
    
    #for i in xrange(rounds):
    for i in xrange(264, rounds):
        # read chunk_size number of predictions from each file
        df = pd.DataFrame()
        chunks = []
        for fname in fnames:
            chunks.append(pd.read_csv(fname, skiprows = skip, nrows = CHUNK_SIZE, names = header))
        
        # aggregate 50 prob forecast to make final prob 
        df = pd.concat(chunks)
        tmp = df.groupby('id', as_index = False)
        df = tmp.aggregate(np.mean)
        
        # write results into submission file
        if i > 0:
            df.to_csv(sub_file, mode = 'a', header = False, index = False)
        else:
            df.to_csv(sub_file, index = False)
        skip = skip + CHUNK_SIZE
        
    # zip the file for susubmission
    with zipfile.ZipFile(sub_zip, 'w', zipfile.ZIP_DEFLATED) as myzip:
        myzip.write(sub_file)

        
