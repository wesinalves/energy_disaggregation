from __future__ import print_function
import sip
import nilmtk
import matplotlib.pyplot as plt
import pandas as pd 

dataset = nilmtk.DataSet('ukdale.h5')

#dataset.set_window(start="6-4-2013")
#dataset.set_window(end="30-1-2013")
#dataset.set_window(start="6-11-2014",end="13-11-2014")

BUILDING = 2

elec = dataset.buildings[BUILDING].elec

gt= {}
sample_period = 6
for i, chunk in enumerate(elec.mains().load(sample_period=sample_period)):
    chunk_drop_na = chunk.dropna()
    gt[i]={}

    for meter in elec.submeters().select_using_appliances(type=['kettle', 'fridge','microwave','dish washer', 'washing machine']).meters:
        # Only use the meters that we trained on (this saves time!)   
        gt[i][meter] = meter.load(sample_period=sample_period).next()
    


    gt[i] = pd.DataFrame({k:v.squeeze() for k,v in gt[i].iteritems()}, index=gt[i].values()[0].index).dropna()