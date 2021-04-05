# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
import os
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
import json
import pickle
from zipfile import ZipFile


# %%
# Compute the note placements in one beat map and return a list of their index in the most common
def get_note_placements_by_index(dat_json, most_common_placements, min_time_slice=0.0625):
    # List of all notes, not grouped with notes at same times
    notes_list = dat_json['_notes']
    # List of all unique time points that notes are at
    note_timings = set([note['_time'] for note in notes_list])
    # Dictonary mapping time point to list of notes at that time. Beat saber has 3x4 grid of note positions (=12)
    # Time points are actually the time in terms of beat number
    notes_at_time_point = {note_timing : [0] * 12 for note_timing in note_timings}
    for note in notes_list:
        # 0 - Red, 1 - Blue
        colour = note['_type'] 
        # If it is a bomb then skip as our model doesn't deal with bombs
        if colour not in [0, 1]:
            continue
        # Direction is direction you must cut the note 
        # 0 - Up, 1 - Down, 2 - Right, 3 - Left,
        # 4 - Down-Right, 5 - Down-Left, 6 - Up-Right, 7 - Up-Left
        # 8 - No Direction
        direction = note['_cutDirection']

        # Integer classification based on colour and direction. (Colour * 9 since 9 directions per colour)
        note_type = colour * 9 + direction + 1 # Plus 1 to account for 0 being no note
        
        # Ranges from 0 to 2 (3x4 grid)
        row = note['_lineLayer']
        # Ranges from 0 to 3 (3x4 grid)
        col = note['_lineIndex']
        # Convert grid location to 1D array location
        grid_index = row * 4 + col
        # Prevent mapping and noodle extensions maps from indexing out of bounds (indexes can be negative in these extenstions)
        if abs(grid_index) > 11: 
            continue # These arent actually notes but something else in mapping extensions
        # Update the dictionary with the location and type of note (convert grid to 1D array location)
        try:
            notes_at_time_point[note['_time']][grid_index] = note_type
        except Exception as e:
            print(e, "row {}, col {}, note {}".format(row, col, note))
    
    # Determine the index of the placements in the song and store in dictonary
    placement_at_time_points = OrderedDict()
    for time_point, placement in sorted(notes_at_time_point.items()):
        placement_tuple = tuple(placement)
        try:
            placement_index = most_common_placements.index(placement_tuple)
        # If the placement is not in the most common then we pretend there is no notes there
        except Exception as e:
            placement_index = 0
        finally:
            placement_at_time_points[time_point] = placement_index

    return placement_at_time_points


# %%
# Compute the features of the notes at a given time point (for the CRF model)
def get_placement_features(dat_json, most_common_placements):
    # List of all notes, not grouped with notes at same times
    notes_list = dat_json['_notes']
    # List of all unique time points that notes are at
    note_timings = set([note['_time'] for note in notes_list])
    # Dictonary mapping time point to dictonary of features (basically anything with a relationship between i and i - 1)
    notes_at_time_point = {note_timing : {'placement' : [0] * 12,
                                          'placement_index' : '0', # Want it to be string since its also the label
                                          'time_point' : "{}".format(note_timing), # This is needed
                                          'time_since_last_note' : '0',
                                          'time_to_next_note' : '0',
                                          'num_notes' : '0',
                                          'colours' : [], 
                                          'rows' : [], 
                                          'cols' : [],
                                          'directions' : [],
                                          'placement_count' : '0',
                                          'prev_placement' : '0', 
                                          'next_placement' : '0' } # Maybe add note number 
                            for note_timing in note_timings}
    for note in notes_list:
        # 0 - Red, 1 - Blue
        colour = note['_type'] 
        if colour not in [0, 1]: # Must be bomb
            continue
        direction = note['_cutDirection']
        note_type = colour * 9 + direction + 1 # Plus 1 to account for 0 being no note   
        row = note['_lineLayer']
        col = note['_lineIndex']
        grid_index = row * 4 + col
        # Prevent mapping and noodle extensions maps from indexing out of bounds (indexes can be negative in these extenstions)
        if abs(grid_index) > 11: 
            continue # These arent actually notes but something else in mapping extensions
        note_time = note['_time']
        try:
            # Update the dictionary with the location and type of note (convert grid to 1D array location)
            notes_at_time_point[note_time]['placement'][grid_index] = note_type
            # Update the features info
            notes_at_time_point[note_time]['num_notes'] = "{}".format(int(notes_at_time_point[note_time]['num_notes']) + 1)
            notes_at_time_point[note_time]['colours'].append("{}".format(colour))
            notes_at_time_point[note_time]['rows'].append("{}".format(row))
            notes_at_time_point[note_time]['cols'].append("{}".format(col))
            notes_at_time_point[note_time]['directions'].append("{}".format(direction))
        except Exception as e:
            print(e, "row {}, col {}, note {}".format(row, col, note))
    
    # Determine the index of the placements in the song and store in feature dictonary
    features_at_time_points = OrderedDict()
    placement_counter = Counter()
    for time_point, features_dict in sorted(notes_at_time_point.items()):
        # print(f'Time {time_point}\n', json.dumps(features_dict, indent=4))
        placement_tuple = tuple(features_dict['placement'])
        try:
            placement_index = most_common_placements.index(placement_tuple)
        # If the placement is not in the most common then we pretend there is no notes there
        except Exception as e:
            placement_index = 0
        finally:
            features_dict['placement_index'] = "{}".format(placement_index)
            placement_counter.update([placement_index])

            features_at_time_points[time_point] = features_dict

    # Add the information on the previous placement index and next placement index to each placement
    num_timings = len(features_at_time_points)
    prev_time, prev_features = 0, {}
    for i, (time_point, features_dict) in enumerate(list(features_at_time_points.items())): # Ordered dict and already sorted
        # Update with the count of the placement type
        features_dict['placement_count'] = "{}".format(placement_counter[features_dict['placement_index']])
        # Can't do the previous element at start
        if i != 0:
            features_dict['time_since_last_note'] = "{}".format(time_point - prev_time)
            features_dict['prev_placement'] = prev_features['placement_index']
        # Can't do the next element at the end
        if i != num_timings - 1:
            next_time, next_features = list(features_at_time_points.items())[i + 1]
            features_dict['time_to_next_note'] = "{}".format(next_time - time_point)
            features_dict['next_placement'] = next_features['placement_index']
        
        prev_time, prev_features = time_point, features_dict

    return features_at_time_points


# %%
# Get a list of strings where each string is just numbers describing the notes at that time point
def get_notes_as_strings(dat_json):
    # List of all notes, not grouped with notes at same times
    notes_list = dat_json['_notes']
    # List of all unique time points that notes are at
    note_timings = set([note['_time'] for note in notes_list])
    # Dictonary mapping time point to dictonary note information
    # Using 9 for no note instead of 0 since 0 stands for red, row 0 etc.
    notes_at_time_point = {note_timing : {'red_count' : 0,
                                          'blue_count' : 0,
                                          'red_colours' : [9, 9],
                                          'red_directions' : [9, 9],
                                          'red_rows' : [9, 9],
                                          'red_cols' : [9, 9],
                                          'blue_colours' : [9, 9],
                                          'blue_directions' : [9, 9],  
                                          'blue_rows' : [9, 9],
                                          'blue_cols' : [9, 9] }
                            for note_timing in note_timings}
    # Dictonary for colour number to colour name
    colours = {0 : 'red', 1 : 'blue'}
    for note in notes_list:
        note_time = note['_time']
        # 0 - Red, 1 - Blue
        colour_num = note['_type'] 
        if colour_num not in [0, 1]: # Must be bomb
            continue
        colour = colours[colour_num]
        # We are only allowing up to 2 blocks of any colour to limit the string size
        colour_count = notes_at_time_point[note_time][f'{colour}_count']
        if colour_count >= 2:
            continue

        direction = note['_cutDirection']
        row = note['_lineLayer']
        col = note['_lineIndex']
        # Prevent mapping and noodle extensions maps from indexing out of bounds (indexes can be negative in these extenstions)
        if row not in [0, 1, 2] or col not in [0, 1, 2, 3] or direction not in list(range(9)):
            continue # These arent actually notes but something else in mapping extensions
        try:
            notes_at_time_point[note_time][f'{colour}_colours'][colour_count] = colour_num
            notes_at_time_point[note_time][f'{colour}_directions'][colour_count] = direction
            notes_at_time_point[note_time][f'{colour}_rows'][colour_count] = row
            notes_at_time_point[note_time][f'{colour}_cols'][colour_count] = col
            notes_at_time_point[note_time][f'{colour}_count'] += 1
        except Exception as e:
            print(e, "row {}, col {}, note {}".format(row, col, note))
    # Now convert these notes at time points to strings
    notes_as_strings_list = []
    for _, notes in sorted(notes_at_time_point.items()):
        # 16 fields. 4 per note and 2x2 notes (2 per colour max)
        notes_as_string = '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(
            notes['red_colours'][0], notes['red_directions'][0], notes['red_rows'][0], notes['red_cols'][0],
            notes['red_colours'][1], notes['red_directions'][1], notes['red_rows'][1], notes['red_cols'][1],
            notes['blue_colours'][0], notes['blue_directions'][0], notes['blue_rows'][0], notes['blue_cols'][0],
            notes['blue_colours'][1], notes['blue_directions'][1], notes['blue_rows'][1], notes['blue_cols'][1]
        )
        notes_as_strings_list.append(notes_as_string)

    return notes_as_strings_list