# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import sys
import os
import numpy as np
import pandas as pd
import json
import pickle
from zipfile import ZipFile
import librosa


# %%
# Gets a list of onset times seperated by atleast min_sep time
def get_onset_times(song_file, min_sep=0.4):
    # Load song 
    y, samp_rate = librosa.load(song_file)
    # Get onset times
    onset_times = librosa.onset.onset_detect(y=y, sr=samp_rate, units='time')
    # Loop over the array backwards and delete any elements which are too close
    del_count = 0
    for i in range(len(onset_times) - 1, 0, -1): 
        if onset_times[i] - onset_times[i - 1] < min_sep:
            # Move one to the average of the two times to preserve as many notes as possible
            avg_time = (onset_times[i] + onset_times[i - 1]) / 2
            onset_times[i] = avg_time
            onset_times = np.delete(onset_times, i - 1)
            del_count += 1
    print("Removed {} onset times for being within {}s of the next note".format(del_count, min_sep))
    return onset_times


