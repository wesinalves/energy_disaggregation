from __future__ import print_function, division
import time
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#matplotlib inline

rcParams['figure.figsize'] = (13, 6)

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation, FHMM

#dividing data in train and tes set
train = DataSet('ukdale.h5')
test = DataSet('ukdale.h5')



#set window of interest. perhaps timestamp is thrwoing an error
train.set_window(start="1-4-2013",end="30-9-2014")
test.set_window(start="16-10-2014",end="30-10-2014") # to change to 5 house# to change to 5 house

#defining the building of interest. to change for more than one buildings
building = 1
train_elec = train.buildings[building].elec
test_elec = test.buildings[building].elec

#selecting top applience. Change to kettle, fridge, washing machine, microwave and dish washer
#top_5_train_elec = train_elec.submeters().select_top_k(k=5)
top_5_train_elec = train_elec.submeters().select_using_appliances(type=['fridge freezer', 'kettle', 'washer dryer','microwave','dish washer'])


#Training and disaggregation

def predict(clf, test_elec, sample_period, timezone):
    pred = {}
    gt= {}

    for i, chunk in enumerate(test_elec.mains().load(ac_type='active', sample_period=sample_period)):
        chunk_drop_na = chunk.dropna()
        pred[i] = clf.disaggregate_chunk(chunk_drop_na)
        gt[i]={}

        for meter in test_elec.submeters().select_using_appliances(type=['fridge freezer', 'kettle','washer dryer','microwave','dish washer']).meters:
            # Only use the meters that we trained on (this saves time!)    
            gt[i][meter] = meter.load(sample_period=sample_period).next()
    
        gt[i] = pd.DataFrame({k:v.squeeze() for k,v in gt[i].iteritems()}, index=gt[i].values()[0].index).dropna()
        
    # If everything can fit in memory
    gt_overall = pd.concat(gt)
    gt_overall.index = gt_overall.index.droplevel()
    pred_overall = pd.concat(pred)
    pred_overall.index = pred_overall.index.droplevel()

    # Having the same order of columns
    gt_overall = gt_overall[pred_overall.columns]
    
    #Intersection of index
    gt_index_utc = gt_overall.index.tz_convert("UTC")
    pred_index_utc = pred_overall.index.tz_convert("UTC")
    common_index_utc = gt_index_utc.intersection(pred_index_utc)
    
    
    common_index_local = common_index_utc.tz_convert(timezone)
    gt_overall = gt_overall.ix[common_index_local]
    pred_overall = pred_overall.ix[common_index_local]
    appliance_labels = [m.label() for m in gt_overall.columns.values]
    gt_overall.columns = appliance_labels
    pred_overall.columns = appliance_labels
    return gt_overall, pred_overall

#Run classifiers CO and FHMM
classifiers = {'CO':CombinatorialOptimisation(), 'FHMM':FHMM()}
predictions = {}
sample_period = 6
for clf_name, clf in classifiers.iteritems():
    print("*"*20)
    print(clf_name)
    print("*" *20)
    clf.train(top_5_train_elec, sample_period=sample_period)
    gt, predictions[clf_name] = predict(clf, test_elec, 6, train.metadata['timezone'])

#Evaluate algorithms by rmse metric
def compute_rmse(gt, pred):
    from sklearn.metrics import mean_squared_error
    rms_error = {}
    for appliance in gt.columns:
        rms_error[appliance] = np.sqrt(mean_squared_error(gt[appliance], pred[appliance]))
    return pd.Series(rms_error)

def compute_mae(gt, pred):
    from sklearn.metrics import mean_absolute_error
    abs_error = {}
    for appliance in gt.columns:
        abs_error[appliance] = mean_absolute_error(gt[appliance], pred[appliance])
    return pd.Series(abs_error)


error = {}
for clf_name in classifiers.keys():
    error[clf_name] = compute_mae(gt, predictions[clf_name])#choose error: compute_mae or compute_rmse

error = pd.DataFrame(error)

print(error)