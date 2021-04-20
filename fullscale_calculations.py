#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to convert dBu to Voltage peak 2 peak, and 
understand if my calculations have all been correct or not. 

Created on Tue Apr 20 12:03:46 2021

@author: Thejasvi Beleyur
"""
import numpy as np 

dbu_ref = 0.775 # volts

def dbu2vrms(dbu_value):
    return dbu_ref*10**(dbu_value/20)

def vrms2vp2p(vrms_value):
    return 2*np.sqrt(2)*vrms_value

def dbu2vp2p(dbu_value):
    vrms = dbu2vrms(dbu_value)
    vp2p = vrms2vp2p(vrms)
    return vp2p


#%%
# The max input levels for the Fireface 802 is different across the 
# front and back channels, also depending on if it's a mic or instrument
# input. 

max_in_level = np.array([10,19,21])

print(dbu2vp2p(max_in_level))