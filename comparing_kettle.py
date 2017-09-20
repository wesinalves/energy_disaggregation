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

import nilmtk

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation, FHMM

#############################prepare training data dividing data in train and tes set
dataset1 = DataSet('ukdale.h5')
dataset2 = DataSet('ukdale.h5')
dataset3 = DataSet('ukdale.h5')
dataset4 = DataSet('ukdale.h5')
dataset5 = DataSet('ukdale.h5')

train_elec = []

dataset1.set_window(start="2013-12-04",end="2015-07-01")
elec1 = dataset1.buildings[1].elec
train_elec.append(elec1.submeters().select_using_appliances(type=['kettle']))

dataset2.set_window(start="2013-05-22",end="2013-10-03")
elec2 = dataset2.buildings[2].elec
train_elec.append(elec2.submeters().select_using_appliances(type=['kettle']))


dataset3.set_window(start="2013-02-27",end="2013-04-01")
elec3 = dataset3.buildings[3].elec
train_elec.append(elec3.submeters().select_using_appliances(type=['kettle']))

dataset4.set_window(start="2013-03-09",end="2013-09-24")
elec4 = dataset4.buildings[4].elec
train_elec.append(elec4.submeters().select_using_appliances(type=['kettle']))


#fridges = nilmtk.global_meter_group.select_using_appliances(type='fridge')
train = []
for item in train_elec:
    for meter in item.meters:
        train.append(meter)

input_meter = MeterGroup(train)
#########################################################

test = DataSet('ukdale.h5')



#set window of interest. perhaps timestamp is thrwoing an error
test.set_window(start="2014-10-16",end="2014-10-30") # to change to 5 house# to change to 5 house

#defining the building of interest. to change for more than one buildings
building = 1

test_elec = test.buildings[building].elec


#Training and disaggregation

def predict(clf, test_elec, sample_period, timezone):
    pred = {}
    gt= {}

    for i, chunk in enumerate(test_elec.mains().load(ac_type='active',sample_period=sample_period)):
        chunk_drop_na = chunk.dropna()
        pred[i] = clf.disaggregate_chunk(chunk_drop_na)
        gt[i]={}

        for meter in test_elec.submeters().select_using_appliances(type=['kettle']).meters:
            # Only use the meters that we trained on (this saves time!)    
            gt[i][meter] = meter.load(ac_type='active',sample_period=sample_period).next()
    
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
    clf.train(input_meter, sample_period=sample_period)
    gt, predictions[clf_name] = predict(clf, test_elec, 6, test.metadata['timezone'])

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