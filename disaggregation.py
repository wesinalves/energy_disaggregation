# import libraries
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
from nilmtk.disaggregate import fhmm_exact

#loading DataSet
train = DataSet('ukdale.h5')
test = DataSet('ukdale.h5')

#choose building and window of observation
building = 2

train.set_window(start="2013-06-01",end="2013-06-30")
test.set_window(start="2013-09-15",end="2013-09-30")

train_elec = train.buildings[building].elec
test_elec = test.buildings[building].elec

#train_elec.plot()

#test_elec.mains().plot()


top_5_train_elec = train_elec.submeters().select_top_k(k=5)
start = time.time()

fhmm = fhmm_exact.FHMM()
# Note that we have given the sample period to downsample the data to 1 minute. 
# If instead of top_5 we wanted to train on all appliance, we would write 
# fhmm.train(train_elec, sample_period=60)
fhmm.train(top_5_train_elec, sample_period=6)
end = time.time()
print("Runtime =", end-start, "seconds.")

pred = {}
gt= {}

for i, chunk in enumerate(test_elec.mains().load(sample_period=6)):
    chunk_drop_na = chunk.dropna()
    pred[i] = fhmm.disaggregate_chunk(chunk_drop_na)

    gt[i]={}
    
    for meter in test_elec.submeters().meters:
        # Only use the meters that we trained on (this saves time!)    
        gt[i][meter] = meter.load(sample_period=6).next()
    gt[i] = pd.DataFrame({k:v.squeeze() for k,v in gt[i].iteritems()}, index=gt[i].values()[0].index).dropna()

#plt.show()