# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:13:36 2022

@author: ndcm1133
"""

# Load modules
# Data handling and analysis modules -
import numpy as np
import math
from scipy import signal, io
import sys
import subprocess
import os
import csv
import pandas as pd
from scipy.signal import spectrogram, periodogram, lfilter, freqz, iirfilter
from scipy.interpolate import interp1d

# Plotting modules
from matplotlib import pyplot as plt

# For representing a datetime
from datetime import datetime
import dateutil.parser as dparser

# For raising warnings
import warnings
import mne

import shutil
import os

from read_Dyneumo_LFP_Format import read_Dyneumo_LFP_Format

def resample_by_Interpolation(input_signal, input_Fs, output_Fs):
    
    scale = output_Fs / input_Fs
    # Calculation new length of sample
    n = int(np.round(len(input_signal)*scale))
    
    # Now use linear interpolation to resample the signal
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False), 
        np.linspace(0.0, 1.0, len(input_signal), endpoint=False), 
        input_signal,
    )
    
    return resampled_signal

def convert_csv_File_to_DyNeuMo_Format(input_filename, output_filepath, output_filename, DyNeuMo_2_Template_csv_Filename, Fs_recorded, Fs_DyNeuMo):
    
    # Read/load the EDF file
    csv_Data = np.loadtxt(input_filename, delimiter=',')
    
    # Set the sampling frequency of the data
    Fs = Fs_recorded
    
    # Check if the output directory should be created
    try:
        # Make the output directory
        os.makedirs(output_filepath)
    except FileExistsError:
        # directory already exists
        pass
    
    # Copy the template of the DyNeuMo-2 Data files
    shutil.copyfile(DyNeuMo_2_Template_csv_Filename, output_filepath + '/' + output_filename)
    
    # Now append the corresponding data to the files
    new_file_name = output_filepath + '/' + output_filename
    with open(new_file_name, 'a') as f:

        # Resample the event snippet to the correct sampling frequency
        data_snippet = resample_by_Interpolation(csv_Data, Fs, Fs_DyNeuMo)
        
        # if upsampling then lowpass filter the upsampled signal
        if Fs < Fs_DyNeuMo:
            
            lowpass_cutoff = Fs + Fs*(0.01)        # low cutoff frequency
            lowpass_filter_order = 4
            lowpass_b, lowpass_a = iir_butter_lowpass(lowpass_cutoff, Fs_DyNeuMo, order=lowpass_filter_order)
            data_snippet = lfilter(lowpass_b, lowpass_a, data_snippet)

        
        # Save the event snippet in the DyNeuMo-2 format
        np.savetxt(f, np.array([np.arange(0,len(data_snippet),1), data_snippet]).T, delimiter=", ")

# Function for generating filter co-efficients
def iir_butter_lowpass(lowcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = iirfilter(order, low, btype='low', analog=False, ftype='butter')
    return b, a



#%% Test writing the csv data to the DyNeuMo format 

input_filename = 'C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/Alpha demo Wk8/alpha_Demo_Raw_Data.csv'
output_filepath = 'C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/Alpha demo Wk8/'
output_filename = 'alpha_Demo_DyNeuMo_Format.csv'

DyNeuMo_2_Template_csv_Filename = "C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/DyNeuMo_Software/DyNeuMo_Pipeline/DyNeuMo_2_Pipeline/DyNeuMo_2_Template_csv_File.csv"
Fs_recorded = 125
Fs_DyNeuMo = 625

convert_csv_File_to_DyNeuMo_Format(input_filename, output_filepath, output_filename, DyNeuMo_2_Template_csv_Filename, Fs_recorded, Fs_DyNeuMo)

#%% Save the alpha demo signal as a test signal for the C# filtering code
input_filename = 'C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/Alpha demo Wk8/alpha_Demo_Raw_Data.csv'

# Read/load the csv file
csv_Data = np.loadtxt(input_filename, delimiter=',')
Fs_recorded = 125
Fs_DyNeuMo = 625
interpolated_signal = resample_by_Interpolation(csv_Data, Fs_recorded, Fs_DyNeuMo)
test_signal_filename = 'C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/raw_test_signal.csv'

# Conversion factor for converting volts to adc steps
variable_gain_factor = 1
adc_conversion_factor = 25e-3/(variable_gain_factor * pow(2,16))

# Convert signal from volts to adc steps - round to closest adc step
converted_signal = np.round(interpolated_signal/adc_conversion_factor)
#rounded_signal = np.round(converted_signal)

plt.figure()

plt.plot(converted_signal)
#plt.plot(rounded_signal)


np.savetxt(test_signal_filename, converted_signal, delimiter=',', fmt='%d')

#%% Compare original versus filtered signal from C#
input_raw_filename = 'C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/raw_test_signal.csv'
input_filtered_filename = 'C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/filtered_test_signal.csv'

# Read/load the csv file
raw_signal = np.loadtxt(input_raw_filename, delimiter=',')
filtered_signal = np.loadtxt(input_filtered_filename, delimiter=',')

# Plot the signals
plt.figure()
plt.plot(raw_signal)
plt.plot(filtered_signal)



#%% Plot the test signal for debugging the Picostim API filtering
input_filename = 'C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/raw_test_signal.csv'

# Read/load the csv file
raw_csv_Data = np.loadtxt(input_filename, delimiter=',')

plt.figure()
plt.plot(raw_csv_Data)

#%% Test writing the csv data to the DyNeuMo format 

input_filename = 'C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/Baker_Data/sleep_Onset_Data.csv'
output_filepath = 'C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/Baker_Data/'
output_filename = 'sleep_Onset_DyNeuMo_Format.csv'

DyNeuMo_2_Template_csv_Filename = "C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/DyNeuMo_Software/DyNeuMo_Pipeline/DyNeuMo_2_Pipeline/DyNeuMo_2_Template_csv_File.csv"
Fs_recorded = 625
Fs_DyNeuMo = 625

convert_csv_File_to_DyNeuMo_Format(input_filename, output_filepath, output_filename, DyNeuMo_2_Template_csv_Filename, Fs_recorded, Fs_DyNeuMo)

#%% Test writing the csv data to the DyNeuMo format 

input_filename = 'C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/Alpha demo Wk8/offset_filtered_alpha_demo.csv'
output_filepath = 'C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/Alpha demo Wk8/'
output_filename = 'alpha_Demo_No_Offset_DyNeuMo_Format.csv'

DyNeuMo_2_Template_csv_Filename = "C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/DyNeuMo_Software/DyNeuMo_Pipeline/DyNeuMo_2_Pipeline/DyNeuMo_2_Template_csv_File.csv"
Fs_recorded = 625
Fs_DyNeuMo = 625

convert_csv_File_to_DyNeuMo_Format(input_filename, output_filepath, output_filename, DyNeuMo_2_Template_csv_Filename, Fs_recorded, Fs_DyNeuMo)


#%%

csv_Data = np.loadtxt(input_filename, delimiter=',')

plt.figure()
plt.plot(csv_Data)

#%% Subprocess debugging
path_to_filter_executable = 'C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/DyNeuMo_Software/DyNeuMo_Pipeline/DyNeuMo_2_Pipeline/Robert_DyNeuMo_Filter_Tool'
filter_tool_executable = 'C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/DyNeuMo_Software/DyNeuMo_Pipeline/DyNeuMo_2_Pipeline/Robert_Filter_Tool/filter_tool.exe'

result = subprocess.run(
    'filter_tool.exe --offset 0.1 --bandpass 4 18 22 --abs --movexp f 1 --lowpass 2 5 --fs 625', capture_output=True
)
print("stdout:", result.stdout)


path_to_filter_executable = 'C:/Users/ndcm1133/OneDrive - Nexus365/Desktop/DyNeuMo_Software/DyNeuMo_Pipeline/DyNeuMo_2_Pipeline/Robert_DyNeuMo_Filter_Tool'
result = subprocess.run(
    path_to_filter_executable+'/filter_tool.exe --offset 0.1 --bandpass 4 18 22 --abs --movexp f 0.1 --lowpass 2 5 --fs 625', capture_output=True
)


import json
import ast

# Convert the subprocess result to a string that can be loaded as json
filter_parameters_string = result.stdout.decode('utf-8')

# Load the json as a python dictionary
filter_parameters = json.loads(filter_parameters_string)

filter_parameters['coeffs']["4"]["movexp_shift"]




import ast
filter_blocks_string = result.stdout.decode('utf-8').split('\r\n\r\n')[0]
filter_coeffs_string = result.stdout.decode('utf-8').split('\r\n\r\n')[1]

# Parse the filter blocks portion of the string
parsed_block_string = filter_blocks_string.split(' = ')
filter_blocks = ast.literal_eval(parsed_block_string[1])
len(filter_blocks)

# Parse the filter coefficients portion of the string
parsed_filter_coeffs_string = filter_coeffs_string.split(' = ')

# Json version for testing
parsed_filter_coeffs_string[1] = '{"0": {"offset_shift": 10}, "1": {"gain": 22289, "a0": 16384, "a1": -31535, "a2": 15893, "b0": 16384, "b1": 0, "b2": -16384, "shift_gain": 5, "shift_a": 14, "shift_b": 14, "shift_za": 0, "shift_return": 15}, "2": {"gain": 19388, "a0": 16384, "a1": -31781, "a2": 15956, "b0": 16384, "b1": 0, "b2": -16384, "shift_gain": 5, "shift_a": 14, "shift_b": 14, "shift_za": 0, "shift_return": 15}, "3": {}, "4": {"movexp_shift": 7}, "5": {"gain": 20463, "a0": 16384, "a1": -31604, "a2": 15260, "b0": 16384, "b1": 32767, "b2": 16384, "shift_gain": 10, "shift_a": 14, "shift_b": 14, "shift_za": 0, "shift_return": 15}}'

import json

filter_coeffs = json.loads(parsed_filter_coeffs_string[1])
len(filter_coeffs["1"])

if 'gain' in filter_coeffs["1"].keys():
    print('Yes!')
else:
    print('No!')



  