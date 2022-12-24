# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 09:32:04 2022

@author: ndcm1133
"""

# Import Required packages ----------------------- Note - recheck are all necessary
import numpy as np
from scipy import signal
import csv

# For representing a datetime
from datetime import datetime
import dateutil.parser as dparser

# For raising warnings
import warnings

def read_Dyneumo_LFP_Format(filename, dyneumo_mark):
  """ Reads DyNeuMo LFP recordings from .csv file.
    
    Parameters
    ----------
    filename : string
        The name of the data file. The data should have been recorded with PicoPC.
    dyneumo_mark : int
        Variable for identifying the version of the DyNeuMo that was used to record the data.
        1 for DyNeuMo-1 and 2 for DyNeuMo-2

    Returns
    -------
    lfp : a dictionary of the data read from the fields in the input file
        Dictionary Entries:
            Fs : int
                sampling rate, derived from the input dyneum_mark
            subject_id : string
                the patient name
            timestamp : datetime
                the recording start date and time
            isotimestamp : 
                same as timestamp but in ISO format
            electrodes : string
                recording electrodes with cathode 'C' and anode 'A'
            t : 1-D numpy array of floats
                the time vector starting from zero
            x : 1-D numpy array of floats
                the LFP signal in V. For DyNeuMo-2 x is 'raw' or 'filtered'
                depending on the 'Options' register at recording time
            xf : 1-D numpy array of floats
                the filtered LFP signal in V. DyNeuMo-2 only, it can be present
                or not. It is the algorithm output calculated by PicoPC. The
                algorithm parameters are from the table in the 'Schedule' tab
                based on read_dyneumo_LFP_format.m

  """
  
  # Dictionary for the loaded LFP signal
  lfp = {}
  
  # Check the version of the DyNeuMo
  if dyneumo_mark == 1:
    Fs = 1000                 # DyNeuMo-1 Sampling Frequency
  elif dyneumo_mark == 2:
    Fs = 625                  # DyNeuMo-2 Sampling Frequency
  else:
    warnings.warn("Input dyneumo_mark is not valid. Must be either 1 or 2")
    return
  
  try:
    # Open file
    with open(filename, "r") as fi:
        lines = fi.readlines()
        lines = [line.rstrip() for line in lines]
        lines = [line.replace("\n", "") for line in lines]
        
        # Get the file id
        id = lines[0]
        
        # Get the timestamp for the file
        tstmp = lines[1]
        
        # Get the electrodes that were used - Need to update
        electrodes = []
        
        # Create lists for the recorded data
        x = []
        t = []
        xf = []
        
        for data_sample_id in np.arange(5, len(lines), 1):
            data_sample = lines[data_sample_id].split(',')
            t.append(int(float(data_sample[0])))
            x.append(float(data_sample[1]))
            
            # Included filtered signal
            if len(data_sample) > 2:
                xf.append(float(data_sample[2]))
        
        # Convert lists to arrays
        t = np.array(t)
        x = np.array(x)
        xf = np.array(xf)
        
        # Convert units 
        t = (t - t[0])/Fs               # sample times into seconds
        x = x                           # recorded signal into volts
        
        # Parse the date and time
        isotimestamp = dparser.parse(tstmp, fuzzy=True)
        isotimestamp = isotimestamp.strftime('%y%m%dT%H%M%S')
        
        # Make dictionary of the LFP properties
        # if there is a filtered signal - 
        if len(data_sample) > 2:
            lfp = {'Fs' : Fs, 'subject_id': id, 'timestamp': tstmp, 'isotimestamp': isotimestamp, 'electrodes': electrodes, 't': t, 'x': x, 'xf': xf }
        # else no filtered signal
        else:
            lfp = {'Fs' : Fs, 'subject_id': id, 'timestamp': tstmp, 'isotimestamp': isotimestamp, 'electrodes': electrodes, 't': t, 'x': x}
            
        # return the lfp dictionary
        return lfp
        
  # Exception if file doesn't exist
  except FileNotFoundError:
    warnings.warn("Can't find " + filename + "file :(")
    return