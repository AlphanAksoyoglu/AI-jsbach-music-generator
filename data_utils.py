'''
This library contains data handlers for reading, and transforming MusicXML
data to tensors suitable for a neural net input

DataIOHAndler -> Handles pickling and unpickling the data
MusicHandler -> Handles MusicXML files, creates voices from score, and encodes the required note sequence in a character
notation format
NeuralNetIOHandler -> Converts the data from MusicHandler into onehot-encoded tensors suitable for Neural Network Input
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

class Colors:
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class DataIOHandler():
    
    '''
    Class that takes care of pickling and unpickling processed data
    '''
    
    def __init__(self, path="./data/"):
        self.path = path 
    
    def pickle_data(self, data, file_name):
        outfile = open(f'{self.path}{file_name}.pickle','wb')
        pickle.dump(data,outfile)
        outfile.close()
    
    def get_pickle_data(self, file_name):
        infile = open(f'{self.path}{file_name}','rb')
        data = pickle.load(infile)
        infile.close()
        return data


class MusicHandler():
    '''
    This class takes care of reading music data in mxml files
    create transpositions to multiply data and decoding them in the format we need
    Usage:
    Only process_data nd create_corpus needs to be externally accessed, self.corpus stores the needed corpus
    '''
    def __init__(self, path, max_parts, transpose=True):
        
        self.path = path
        self.max_parts = max_parts
        self.transpose = transpose
        self.named_data = None
        self.corpus = []
        self.major_transpose_list = ['C','D','G','A','E','B','F','B-','E-','A-','C#','F#','G-','D-']
    def process_data(self):
        
        if self.transpose:
            print(f'I will read the files at {self.path} and also transpose them')
            self.named_data = self._get_data_name(self.path)
            self.named_data = self._expand_data_with_transposition(self.named_data)
        else:
            print(f'I will read the files at {self.path} and will not transpose them')
            self.named_data = self._get_data_name(self.path)
            
    def create_corpus(self):
    
        for i in self.named_data:
            print( f"Reading {i[0]} into Corpus")
            
            try:
                self.corpus.append((i[0],self._get_voices(i[1])))
            except:
                print(f'Could not process {i[0]}, possible bad score')
                continue
        print( f"Done")
        
    def _get_data_name(self, path: str) -> List[Tuple(str,Type[music21.stream.Score])]:

        '''
        Reads music xml files and returns a list of tuples with (filename, music21.stream.object)
        '''

        files = []
        for filename in os.listdir(path):
            if filename.endswith(".xml"):
                print(f"Reading {filename}")
                files.append((filename,converter.parse(f'{path}/'+filename)))
        print( f"Done")
        return files

    def _expand_data_with_transposition(self, data: List[Tuple(str,Type[music21.stream.Score])]) -> List[Tuple(str,Type[music21.stream.Score])]:
        '''
        Gets output from get_data_with_name and creates transposed scores
        '''
        extra_data = []
        for item in data:
            print(f"Transposing {item[0]}")
            extra_data = extra_data + self.create_transposed_scores(item)
        data = data + extra_data
        print(f"Finished Transposing")
        return data
    
    def create_transposed_scores(self, music_data: List[Tuple(str,Type[music21.stream.Score])]) -> List[Tuple(str,Type[music21.stream.Score])]:
        '''
        Creates transposed scores from an mstream
        '''

        transposed_scores = []

        mstream = music_data[1]
        mstream_keys = mstream.flat.getKeySignatures()
        current_key = mstream_keys[0].tonic

        to_keys = self.get_transpose_list(current_key.name)


        for to_key in to_keys:

            key_interval = interval.Interval(current_key, pitch.Pitch(to_key))
            mstream_transposed = mstream.transpose(key_interval)
            item = (f'{music_data[0]}_transposed_to_{to_key}',mstream_transposed)
            transposed_scores.append(item)

        return transposed_scores

    def get_transpose_list(self, my_key):

        '''

        '''

        if (my_key not in ['C#','F#','G-','D-']):
            transposes = [i for i in self.major_transpose_list if i != my_key] 
            transposes.remove('D-')
            transposes.remove('G-')
        elif (my_key in ['C#','D-']):
            transposes = [i for i in self.major_transpose_list]
            transposes.remove('C#')
            transposes.remove('D-')
            transposes.remove('G-')
        elif (my_key in ['F#','G-']):
            transposes = [i for i in self.major_transpose_list]
            transposes.remove('F#')
            transposes.remove('G-')
            transposes.remove('D-')

        return transposes
    
    def _get_voices(self, mstream: music21.stream.Score) -> np.array(List[str]):
    
        '''
        Reads a music21 stream object and splits the score into MAX_PARTS of voices
        Gaps in the note sequence is filled with rests
        The note sequence of each voice is extracted, fastest note being 1/32nd notes
        Rests are indicated as '_'
        Returns an Array of MAX_PARTS + 3 Lists (Musical Key, Start Sequence, End Sequence)
        '''

        d_pitch = [] #pitches will be stored here
        score_key = mstream.analyze('key').name #musical key

        this_stream = mstream.voicesToParts() #split score into parts
        number_of_parts = len(this_stream) #get total number of voices

        #if number of parts < MAX_PARTS the missing part will be composed of rests
        missing_parts = self.max_parts - number_of_parts 

        #fill the gaps in timespace with rests
        for i in this_stream.parts:
            i.makeRests(fillGaps=True,inPlace=True)

        #for each part collect the pitches and append to pitch array    
        for i in range(this_stream.parts.elementsLength):

            pitches = []

            this_stream.parts[i].makeRests(fillGaps=True,inPlace=True)

            #flatted the voices here, we might still catch some chords
            for element in this_stream.parts[i].flat:

                #handle notes
                if isinstance(element, note.Note):
                    if (element.quarterLength != 0.0):
                        pitches.append(element.pitch.nameWithOctave)
                        for i in range (int(element.quarterLength/0.125) - 1):
                            pitches.append('_')

                #handle_chords
                elif isinstance(element, chord.Chord):
                    if (element.quarterLength != 0.0):
                        pitches.append(element[-1].pitch.nameWithOctave)
                        for i in range (int(element.quarterLength/0.125) - 1):
                            pitches.append('_')

                #handle rests
                elif isinstance(element, note.Rest):
                    pitches.append('rest')
                    for i in range (int(element.quarterLength/0.125) - 1):
                            pitches.append('_')


            d_pitch.append(pitches.copy()) 

        #if the music ends with sequence of rests, music21 does not fill it with rests, we fill them with rests here
        d_pitch_maxes = np.array([len(i) for i in d_pitch])

        for i in d_pitch:
            if len(i)<d_pitch_maxes.max():
                while len(i) < d_pitch_maxes.max():
                    i.append('_')


        length_score = len(d_pitch[0])

        #these vectors indicate the key, start and end sequence of the score
        key_vector = np.array([f'{score_key}']*length_score)
        start_vector = np.array([0]*length_score)
        end_vector = np.array([0]*length_score)

        start_vector[0] = 1
        end_vector[-1] = 1

        #fill missing parts with rests
        if(missing_parts):
            missing_vector = np.array(["_"]*length_score)


        for i in range(missing_parts):
            d_pitch = np.vstack((d_pitch, missing_vector))

        #finally vstack all return the array
        d_pitch = np.vstack((d_pitch,key_vector,start_vector,end_vector))

        return np.array(d_pitch)

class NeuralNetIOHandler():
    
    '''
    This class creates the sparse matrix one hot encoded inputs, and one hot encoded outputs for feeding into the neural net
    Usage:
    create_corpus_array_numeric() : Encoded data received by the MusicHandler is numerically encoded
    
    
    create_onehot_inputs():
    create_onehot_outputs():
    This data is read for unique data points and categorized, and the categorized data is one hot encoded and put in a
    shape suitable for the neural network
    '''
    
    def __init__(self, corpus):
        
        self.input_corpus = corpus
        self.corpus_numeric = []
        self.corpus = self._join_corpus()
        self.inputs = None
        self.outputs = []
        self.one_hot_outputs = None
        self.vocabs = {}
        self.categories = None
    
    def _join_corpus(self):
        print("Joining Corpus...")
        
        corpus_join = []
        
        for i in self.input_corpus:
            corpus_join.append(i[1])
        print("Done")
        return np.hstack(corpus_join)
        
    def _create_vocabs(self):
        
        '''
        Creates a vocabulary to convert text corpus to a numerically encoded corpus
        '''

        soprano_pitches = sorted(set(_ for _ in self.corpus[0]))
        soprano_vocab   = dict((note, number) for number, note in enumerate(soprano_pitches))
        
        alto_pitches    = sorted(set(_ for _ in self.corpus[1]))
        alto_vocab      = dict((note, number) for number, note in enumerate(alto_pitches))
        
        tenor_pitches   = sorted(set(_ for _ in self.corpus[2]))
        tenor_vocab     = dict((note, number) for number, note in enumerate(tenor_pitches))
        
        bass_pitches    = sorted(set(_ for _ in self.corpus[3]))
        bass_vocab      = dict((note, number) for number, note in enumerate(bass_pitches))
        
        bass_2_pitches  = sorted(set(_ for _ in self.corpus[4]))
        bass_2_vocab    = dict((note, number) for number, note in enumerate(bass_2_pitches))
        
        keys            = sorted(set(_ for _ in self.corpus[5]))
        keys_vocab      = dict((note, number) for number, note in enumerate(keys))
        starts          = sorted(set(_ for _ in self.corpus[6]))
        starts_vocab    = dict((note, number) for number, note in enumerate(starts))
        ends            = sorted(set(_ for _ in self.corpus[7]))
        ends_vocab      = dict((note, number) for number, note in enumerate(ends))

        vocabs_dict = {
            'soprano' : soprano_vocab,
            'alto' : alto_vocab,
            'tenor' : tenor_vocab,
            'bass':bass_vocab,
            'bass_2':bass_2_vocab,
            'keys':keys_vocab,
            'starts':starts_vocab,
            'ends':ends_vocab,
            }

        return vocabs_dict
    
    def create_corpus_array_numeric(self):
        
        '''
        Converts the text based corpus to a numerical corpus
        '''
    
        self.corpus_numeric = []
    
        self.vocabs = self._create_vocabs()

        for _ , (key, value) in enumerate(self.vocabs.items()):
                self.corpus_numeric.append([value[item] for item in self.corpus[_]])  
                
    def create_onehot_inputs(self):
        
        '''
        Creates one-hot encoded inputs of a given sequence length to feed into the neural network
        '''
    
        inputs = []
        outputs = []

        self.categories = [(np.unique(i)) for i in self.corpus_numeric]

        corpus_numeric_array = np.array(self.corpus_numeric).T
        sequence_length = 512
        total_length = corpus_numeric_array.shape[0]


        encoder = OneHotEncoder(categories=self.categories,sparse=True,dtype='uint8')
        #print(cats)

        #inputs = np.zeros((512,605),dtype='uint8')
        #inputs = csr_matrix(inputs)

        for i in range(0, total_length - sequence_length, 1):

            print(f'Currently on: {i}', end="\r", flush=True)

            sequence_in = corpus_numeric_array[i:i + sequence_length]
            #print(sequence_in)
            sequence_out = corpus_numeric_array[i + sequence_length]

            sequence_in = encoder.fit_transform(np.array(sequence_in, dtype='uint8'))
            #sequence_in = sequence_in.reshape(1,512,605)
            #print(sequence_in.shape)
            inputs.append(sequence_in)
            #inputs = np.vstack((inputs,sequence_in))
            #outputs.append(sequence_out)

            #inputs = np.array(inputs)
            #inputs = inputs.inputs.max()
       # outputs = one_hot_encode(np.array(outputs).T)



        self.inputs =  np.array(inputs)#, np.array(outputs)
        

    def create_onehot_outputs(self):
        '''
        This will be merged with create_one_hot_inputs
        '''
    
        inputs = []
        outputs = []

        corpus_numeric_array = np.array(self.corpus_numeric).T
        sequence_length = 512
        total_length = corpus_numeric_array.shape[0]


        for i in range(0, total_length - sequence_length, 1):
            sequence_in = corpus_numeric_array[i:i + sequence_length]
            sequence_out = corpus_numeric_array[i + sequence_length]
            inputs.append(sequence_in)
            outputs.append(sequence_out)

        inputs = np.array(inputs)
        inputs = inputs/inputs.max()
        self.one_hot_outputs = self._one_hot_encode(np.array(outputs).T)
        
        print("Making singular outputs")
        self._output_to_singular()
                
    
    
    def _one_hot_encode(self, data):
        
        '''
        This will be obsoleted when create_one_hot_outputs is merged with create_one_hot_inputs
        '''
    
        categoricals = tuple(to_categorical(i) for i in data)
        return np.hstack(categoricals)
    
    def _output_to_singular(self):
        
        '''
        Splits the output data to seperate outputs corresponding to each voice
        '''

        cats_arr_cumsum = np.cumsum(np.array([len(i) for i in self.categories]))
        print(f'THERE ARE {cats_arr_cumsum[-1]} CATEGORIES SET DENSE LAYER ACCORDINGLY')

        j=0
        for i in cats_arr_cumsum:
            self.outputs.append(self.one_hot_outputs[:,j:i])
            j=i