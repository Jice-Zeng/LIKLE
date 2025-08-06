#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:44:53 2024

@author: zeng623
"""

# Python 3.10, tf 2.15
import os
os.getcwd()
os.chdir('/Users/zeng623/Library/CloudStorage/OneDrive-PNNL/Desktop/Github_Freybery') # Navigate 
import numpy as np
import tensorflow as tf
import bayesflow.diagnostics as diag
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import InvariantNetwork, InvertibleNetwork, SequentialNetwork
from bayesflow.simulation import GenerativeModel, Prior, Simulator
from bayesflow.trainers import Trainer
from sklearn import preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings
from matplotlib.ticker import FormatStrFormatter
warnings.filterwarnings("ignore")

tf.__version__

# %%  training data
##################### Import training data and test data #####################
##############################################################################
NT=4800
reduced_y = np.load('data/New_reduced_Y_const.npy') # this is new data
train_para= reduced_y[:NT,:]  # reduced from 2121 to 631 [344,61,226]
para_scaled=train_para

# 13 observation locations in practice 
# index=np.array([ 49, 55, 68, 178, 247, 285, 377, 401, 430, 472, 541, 610, 630])
head=np.load('data/New_hydraulic_partial.npy') # 25x39x5001
head_t=np.zeros([5001,25,39])
for i in range(5001):
    head_t[i,:,:]=head[:,:,i] # 25 time steps, 3 layers, 706 locations
head_tt=head_t

train_data=head_tt[:NT,:,:]

train_data_t=np.reshape(train_data,[NT*25,39])
scaler_trainO = preprocessing.StandardScaler().fit(train_data_t)
pickle.dump(scaler_trainO, open('scaler_trainO.pkl','wb'))
data_scaled = scaler_trainO.transform(train_data_t)
data_scaled = np.reshape(data_scaled,[NT,25,39])

train_k=tf.convert_to_tensor(para_scaled, dtype=tf.float32) 
train_freqq=tf.convert_to_tensor(data_scaled, dtype=tf.float32) 

# %% vaidation data
val_para= reduced_y[NT:(NT+100),:]   
val_data=head_tt[NT:(NT+100),:,:]

val_para_s= val_para 

val_data_t=np.reshape(val_data,[100*25,39])
val_data_s=scaler_trainO.transform(val_data_t)
val_data_s =  np.reshape(val_data_s,[100,25,39])

val_k=tf.convert_to_tensor(val_para_s, dtype=tf.float32) 
val_freqq=tf.convert_to_tensor(val_data_s, dtype=tf.float32) 

# %% test data
N_test=101
test_para=np.vstack([reduced_y[(NT+100):(NT+202),:],np.zeros([1,631])])  
test_data=head_tt[(NT+100):(NT+202),:,:]
        
test_para_s= test_para 
test_data=test_data
test_data_t=np.reshape(test_data,[N_test*25,39])
test_data_s=scaler_trainO.transform(test_data_t)
test_data_s =  np.reshape(test_data_s,[N_test,25,39])

test_k=tf.convert_to_tensor(test_para_s, dtype=tf.float32) 
test_freqq=tf.convert_to_tensor(test_data_s, dtype=tf.float32)

# %%
###############################################################################
############################## Gnerative Model ################################
###############################################################################
class GenerativeModel:
    def __init__(self, d, random_state=42):

        self.d = d
        self.cntTrain=0
        self.cntTest=0
        self.cntVal=0

    def __call__(self, batch_size):
        return self.generate_datasets(batch_size)
    
    def generate_datasets(self, batch_size): # training data

        if(self.cntTrain+batch_size>train_freqq.shape[0]):
            self.cntTrain=0 
        frequency =train_freqq[self.cntTrain:self.cntTrain+batch_size,:,:]
        theta = train_k[self.cntTrain:self.cntTrain+batch_size,:]
        self.cntTrain=self.cntTrain+batch_size 
        return frequency, theta
    
    def generate_val(self, batch_size): # Test data
        
        if(self.cntVal+batch_size>val_freqq.shape[0]):
            self.cntVal=0 
        frequency_val = val_freqq[self.cntVal:self.cntVal+batch_size,:,:]
        theta_val = val_k[self.cntVal:self.cntVal+batch_size,:]
        self.cntVal=self.cntVal+batch_size
        return frequency_val,theta_val   
    
    def generate_test(self, batch_size): # Test data
        
        if(self.cntTest+batch_size>test_freqq.shape[0]):
            self.cntTest=0 
        frequency_test = test_freqq[self.cntTest:self.cntTest+batch_size,:,:]
        theta_test = test_k[self.cntTest:self.cntTest+batch_size,:]
        self.cntTest=self.cntTest+batch_size
        return frequency_test,theta_test    
    
theta_dim=631
params_names = [r'$\theta_{}$'.format({i+1}) for i in range(theta_dim)]
Freyberg_3D = GenerativeModel(theta_dim)
batch_size=120
n_test=101
####################################################################################################
############################## Define neural approximator ##########################################
####################################################################################################

class ConvNet1D(tf.keras.Model):
    
    def __init__(self):
        super(ConvNet1D, self).__init__()
        
        self.summary_part = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, kernel_size=5, strides=1, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(128, kernel_size=3, strides=1, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(256, kernel_size=3, strides=1, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(768, kernel_initializer='glorot_uniform', activation='elu'),
            
        ])
        
    def call(self, x, training=True):
        x = self.summary_part(x, training=training)
        return x
    
coupling_settings = {
    "affine": {
        "dense_args": dict(
            units=128,
            activation="tanh",
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        ),
        "dropout": True,
        "dropout_prob": 0.5,
    },
    "spline": {
        "dense_args": dict(
            units=128,
            activation="tanh",
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        ),
        "dropout": True,
        "dropout_prob": 0.5,
    },
}

summary_net = ConvNet1D()

inference_net = InvertibleNetwork(num_params=631, num_coupling_layers=12,coupling_design='interleaved',
                                  coupling_settings=coupling_settings)
amortizer = AmortizedPosterior(inference_net, summary_net, name="Frey3D_amortizer")
sim,theta = Freyberg_3D(batch_size) 
# %% load from checkpoint

def configure_input_train(batch_size):
    out_dict = {}
    sim_data,prior_draws = Freyberg_3D(120)
    logdata = sim_data
    params = prior_draws
    out_dict["summary_conditions"] = logdata
    out_dict["parameters"] = params # model parameters
    return out_dict

# configurator is used to connect data with amortizer
checkpoint_path="model_checkpoints/Frey3D_New_reduced_const_12layers"
trainer = Trainer(amortizer=amortizer,generative_model=Freyberg_3D,configurator=configure_input_train,
                  checkpoint_path=checkpoint_path,max_to_keep=1)
# Validation data
def configure_input_valid(n_test):
    out_dict = {}
    sim_data,prior_draws = Freyberg_3D.generate_val(n_test)
    logdata = sim_data
    params = prior_draws
    out_dict["summary_conditions"] = logdata
    out_dict["parameters"] = params

    return out_dict
validation_sims = configure_input_valid(n_test)

# test data
def configure_input_test(n_test):
    data=[]
    sim_data,prior_draws = Freyberg_3D.generate_test(n_test)
    
    for i in range(101):
        out_dict = {}
        logdata = sim_data[i,:,:] 
        params = prior_draws[i,:]

        out_dict["summary_conditions"] = tf.reshape(logdata,[1,25,39])
        out_dict["parameters"] = params
        data.append(out_dict)
    return data
test_sims = configure_input_test(n_test)

def configure_test(n_test):
    out_dict = {}
    sim_data,prior_draws = Freyberg_3D.generate_test(n_test)
    logdata = sim_data

    params = prior_draws

    out_dict["summary_conditions"] = logdata
    out_dict["parameters"] = params

    return out_dict
test_obs = configure_test(n_test)

amortizer.summary()

## Ready to train
h = trainer.train_online(epochs=15000, iterations_per_epoch=40, batch_size=120,validation_sims=validation_sims,
                         early_stopping=True,early_stopping_args={'patience':1000})
# %%
#################################################################################################
######################################### visualization #########################################
#################################################################################################
post_samples=[]
for i in range (101):
    post = amortizer.sample(test_sims[i], n_samples=2000)
    post_samples.append(post)
post_samples=np.array(post_samples)

theta_dim=631
params_names = [r'$\xi_{}$'.format({i+1}) for i in range(theta_dim)]
f = diag.plot_recovery(post_samples[:,:,0:10], test_obs["parameters"][:,0:10], param_names=params_names[0:10])

# # # compute posterior mean and std
samples_mean = tf.reduce_mean(post_samples, axis=1).numpy()
samples_std = tf.math.reduce_std(post_samples, axis=1).numpy()

