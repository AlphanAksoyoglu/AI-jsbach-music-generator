'''
Reads the ./data /p5t4 folder and processes the music there so that it could be fed into the neural network 
'''

from data_utils import *

if __name__ == "__main__":
    
    print("Init Data Handler...")
    data_io_handler = DataIOHandler()
    
    print("Processing music data...")
    music_data = MusicHandler('./data/p5t4', 5, True)
    music_data.process_data()
    music_data.create_corpus()
    
    print("Init Neural Net I/O Handler...")
    nn = NeuralNetIOHandler(music_data.corpus)

    print("Processing Inputs and Outpus...")
    nn.create_corpus_array_numeric()
    nn.create_onehot_inputs()
    nn.create_onehot_outputs()

    print("Writing Neural Net input output data to disk")
    data_io_handler.pickle_data(nn.inputs, 'p5t4_ins')
    data_io_handler.pickle_data(nn.outputs, 'p5t4_outs')
