'''
This library contains helper functions to diagnose issues
Most importantly has the write_score function which converts one_hot encoded scores to back to musical scores
The functions here needs rework...
'''

from __future__ import annotations
from music21 import note, key, chord, stream, converter, interval, pitch
import numpy as np
import pandas as pd
import os
import pickle
import shutil
import fractions
from itertools import groupby
from random import shuffle

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import OneHotEncoder

from scipy import hstack, vstack
from scipy.sparse import csr_matrix

from typing import List, Tuple, Type

def count_parts(mstream):
    a = mstream.voicesToParts(separateById=False)
    return a.parts.elementsLength

def min_qlength(mstream):
    lengths = []
    stream_parted = mstream.voicesToParts()
    for i in range(stream_parted.parts.elementsLength):
         for element in stream_parted.parts[i].flat:
            if isinstance(element, note.Note):
                if (element.quarterLength != 0.0):
                    lengths.append(element.quarterLength)
    return lengths

def decode_from_named(voice_n):
    
    voice_n_list = list(voice_n)
    
    count_dups = [sum(1 for _ in group if _ == '_')  for _, group in groupby(voice_n_list)]
    ordered_dups = [i+1 for i in count_dups if i != 0]

    ordered_notes = [i for i in voice_n_list if i != '_']

    note_dur = [(i,j) for i, j in zip(ordered_notes,ordered_dups)]
    
    return note_dur

def voice_to_part(part,note_dur):
    for i in note_dur:
        if (i[0] != 'rest'):
            n = note.Note(i[0],quarterLength=(i[1]*0.125))
        else:
            n = note.Rest(quarterLength=(i[1]*0.125))
        part.append(n)
        
    return part

def num_parts(mstream):
    return mstream.voicesToParts().parts.elementsLength

def write_score(char_enc_voice_stream):
    
    voices_all = char_enc_voice_stream[:MAX_PARTS,:]
    
    key_name = char_enc_voice_stream[5][0]
    key_pitch = key_name.split()[0]
    key_mode = key_name.split()[1]
    ks = key.KeySignature(key.Key(key_pitch, key_mode).sharps)
    
    number_of_parts = voices_all.shape[0]
    number_of_blank_parts = MAX_PARTS - number_of_parts
    
    g_clef_parts = voice_split_dict[MAX_PARTS][0]
    f_clef_parts = voice_split_dict[MAX_PARTS][1]

    full_score = stream.Score()
    
    c1 = clef.TrebleClef()
    c2 = clef.BassClef()
    c1.offset = 0.0
    c2.offset = 0.0
    
    parts = [stream.Part() for i in range(MAX_PARTS)]
    
    for i in range(MAX_PARTS):
        parts[i].id = f'part{i+1}'
        if i<g_clef_parts:
            parts[i].insert(c1)
        else:
            parts[i].insert(c2)
    
    for i,j in zip(parts,voices_all):
        
        i = voice_to_part(i,decode_from_named(j))
        
#     blank_part = ['_' for i in range(voices_all.shape[1])]
#     blank_part[0] = 'rest'

#     for i in range(number_of_blank_parts):
#         parts.append(blank_part)
    
    for i in parts:
        full_score.insert(i)
        
    piano_score = stream.Score()
    stream_1 = stream.Score()
    stream_2 = stream.Score()
    
    
    
    for i in range(g_clef_parts):
        stream_1.insert(parts[i])
    
    for i in range(f_clef_parts):
        stream_2.insert(parts[i+g_clef_parts])
    
    stream_1 = stream_1.chordify()
    stream_2 = stream_2.chordify()
    
    piano_score.insert(stream_1)
    piano_score.insert(stream_2)
    
    piano_score.parts[0].insert(0,key.Key(key_pitch, key_mode))
    piano_score.parts[0].insert(0,key.Key(key_pitch, key_mode))
    
    #full_score.insert(0,ks)
    
        
    return full_score, piano_score  