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
from collections import OrderedDict
import librosa
import sklearn
from sklearn.model_selection import train_test_split
import markovify as mk

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"\\Map_Processing")
    sys.path.append(module_path+"\\Note_Timing")

from analyze_notes import get_notes_as_strings 
from onset_detection import get_onset_times


# %%
# Returns file path to folder containing all files needed to play song made by model
def get_map_from_song(song_file, markov_chain, model_path='markov_model_64_state.json', output_file_path='Generated_Maps/Expert.dat', start_time=2, bpm=0):
    if markov_chain is None:
        print("Loading Markov model from file")
        with open(model_path, 'r') as markov_file:
            markov_chain_str = json.load(markov_file)
            markov_chain = mk.Chain.from_json(markov_chain_str)

    # Get the onset times where we will place notes
    onset_times = get_onset_times(song_file, min_sep=0.1)
    num_before = len(onset_times)
    onset_times = np.delete(onset_times, np.where(onset_times <= start_time))
    # print("Removed {} onset times for being before the specified start time".format(num_before - len(onset_times)))
    # If the bpm is not provided then we calculate it ourselves
    if bpm == 0:
        y, samp_rate = librosa.load(song_file)
        bpm = librosa.beat.tempo(y=y, sr=samp_rate)
        print("Got a bpm of {}".format(bpm))
    # Determine the notes we should place
    notes_list = markov_chain.walk()
    while len(notes_list) < len(onset_times):
        notes_list = markov_chain.walk()
    
    # Create dictionary with time key and notes values
    notes_at_times = OrderedDict(zip(onset_times, notes_list))
    notes_as_json = convert_notes_string_to_valid_json(notes_at_times, bpm)
    with open(output_file_path, 'w') as dat_file:
        dat_data = {"_version": "2.2.0",
                    "_customData": {
                        "_time": '',
                        "_BPMChanges": [],
                        "_bookmarks": []
                        },
                    "_events": [],
                    "_notes": notes_as_json,
                    "_obstacles": [],
                    "_waypoints": []
                    }
        json.dump(dat_data, dat_file)
    
    # print("Number of notes placed: {}\nNumber of unique note placements: {}\nApprox. notes per second: {}".format(
    #         len(notes_as_json),
    #         len(set(notes_list)),
    #         len(notes_as_json) / np.amax(onset_times)
    #         )
    #     )


# %%
# Takes in ordered dictonary mapping time to notes string and returns list of json
def convert_notes_string_to_valid_json(notes_at_times, bpm):
    list_of_jsons = []
    for time_point, notes_string in notes_at_times.items():
        notes_list = [int(x) for x in notes_string.split(',')]
        assert len(notes_list) == 16
        # Go over the 4 notes in the list and 
        for note_num in range(4):
            try:
                note_info = notes_list[4 * note_num : 4 * (note_num + 1)]
                if note_info[0] not in [0, 1]: # No note
                    continue
                colour = note_info[0]
                direction = note_info[1]
                row = note_info[2]
                col = note_info[3]
                note_json = {"_time": (time_point / 60) * bpm, # Convert to beat timing
                            "_lineIndex": col,
                            "_lineLayer": row,
                            "_type": colour,
                            "_cutDirection": direction}
                list_of_jsons.append(note_json)
            except Exception as e:
                print(e, "note_num {}, max index {}".format(note_num, 4 * (note_num + 1)))
    return list_of_jsons
