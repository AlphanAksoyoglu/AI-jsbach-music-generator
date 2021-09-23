# An LSTM Network for Generating Music Similar to Keyboard Works of J.S Bach

This is a 2-week project I have undertaken as a final project in my bootcamp. Although the pipeline works as intented and it generates music, it needs quite a bit of rework, restructuring along with a better dataset to generate music as intended See: [Future Updates](#future-updates)

To read a detailed description please see section [Detailed Presentation](#detailed-presentation)

To have a test run at its current stage please see [How to Run](#how-to-run)

- [Table of Contents]()
  - [Tech Stack](#tech-stack)
  - [Description](#description)
  - [Data, Transformation, and Network](#data-transformation-and-network)
  - [How to Run](#how-to-run)
  - [Future Updates](#future-updates)
  - [Detailed Presentation](#detailed-presentation)

## Tech Stack

- Python 3.8 (see requirements.tx for libraries)
- TensorFlow
- Keras

## Description

Project mostly inspired by [DeepBach](https://www.flow-machines.com/history/projects/deepbach-polyphonic-music-generation-bach-chorales/) by Sony and [BachBot](https://github.com/feynmanliang/bachbot) by Feynman Liang. Both of these projects used Bach chorales in 4 voices to train their network, which in turn would generate 4 voice chorales by itself.

We can use the same logic to train a network that learns Bach keyboard music (piano, harpsichord, and organ). 

A great overview of several techniques used in music generation is described in this article [Deep Learning Techniques for Music Generation -- A Survey](https://arxiv.org/abs/1709.01620) by Jean-Pierre Briot, Gaëtan Hadjeres, François-David Pachet. They also address the tecnique used by [DeepBach](https://www.flow-machines.com/history/projects/deepbach-polyphonic-music-generation-bach-chorales/) and [BachBot](https://github.com/feynmanliang/bachbot)


## Data, Transformation, and Network

### Data

![Keyboard to Chorale](https://github.com/AlphanAksoyoglu/AI-jsbach-music-generator/blob/main/images/picture1.png?raw=true)


We use music data stored in MusicXML files. The files are obtained from [Kunst der Fuge](www.kunstderfuge.com) and [Tobi's Notenarchiv](https://tobis-notenarchiv.de/wp/).

The music stored here is keyboard or organ music, and normally has 2-3 partitions with polyphonic sequences. These polyphonic sequences in 2-3 partitions should be converted to 4 monophonic voices if we are to follow the recipe set out by [DeepBach](https://www.flow-machines.com/history/projects/deepbach-polyphonic-music-generation-bach-chorales/) and [BachBot](https://github.com/feynmanliang/bachbot)

We perform data augmentation by transposing all the music available to us to different keys In its current stage we ended up with **33 suitable scores** and their transpositions to 12 keys

**We chose music that is separable to a maximum of 5 voices with a time signature of 4/4** 


### Transformation
Image From: [Deep Learning Techniques for Music Generation -- A Survey](https://arxiv.org/abs/1709.01620)
![Encoding](https://github.com/AlphanAksoyoglu/AI-jsbach-music-generator/blob/main/images/picture2.png?raw=true)


We choose to encode the data as done by [DeepBach](https://www.flow-machines.com/history/projects/deepbach-polyphonic-music-generation-bach-chorales/)

We use the MusicXML library to read the music data into Music21 stream object, split this music into 4 voices (with a resolution up to 1/32nd notes) and encode the data. One music data has 7 components. 4 monophonic voices, 1 musical key, and 2 for start and stop sequences

The encoded data is converted to numerical data, then categorized, and finally one hot encoded to feed into the network

This portion of the work is handled by MusicHandler() and NeuralNetIOHandler() classes in data_utils.py

### Network

The Neural Network consists of an input layer, 3 LSTM layers of size 256 (512 in the diagram), and a Dense layer corresponding to the output layer.

![Neural Network](https://github.com/AlphanAksoyoglu/AI-jsbach-music-generator/blob/main/images/picture3.png?raw=true)

## How to run 

The data folder contains 3 example .xml files, only this data will be processed. Processed data will be stored as .pickle files in the data folder and the network will run on this data

Install requirements: `pip install -r requirements.txt` Note: please install gpu version of tensorflow

Process the data: `python generate_nn_data.py`

Run the network for training: `python run_onehot_model.py`


## Future Updates

### Encoding and Data

- Properly 4 voice encoded music acquired from www.kunstderfuge.com will be used for data preparation, manual handling and splitting music into voices causes a lot of issues, a major issue being an overinflated feature space.

- Interpretation and implementations of rests will be revised. An overabundance of rests causes the network to learn to place rests everywhere. Note encoding could be done in the style of [BachBot](https://github.com/feynmanliang/bachbot). This would cause a 2x increase in note feature space but reduce the emphasis on rests

- A resolution down to 1/32nd notes also causes an abundance of rests

### Neural Network Implementation

- One hot encoding scheme could be replaced by a numerical encoding scheme

- The issue of the output space is a complicated matter. I am not sure if the separated output space works as intended. The separated outputs could instead be reduced to a single multi-onehot-encoded vector. Or voice outputs could be a single multi-onehot-encoded and metadata parameters another one

- Optimal parameters should be scanned. Particularly number of LSTM layers and LSTM layer sizes

### Code Cleanup

- Although the forward flow of the data is sufficiently put into class structures, reverse encoding the output from the model is another matter. Mostly functions in the other_utils.py should be put into proper classes

- The music generating script that uses the trained model resides in jupyter notebook, it should be put into a proper script

- Function names should better describe and align with the type of data they get as an input and data they output

- Complete Type annotations would be very useful

## Detailed Presentation

![Slide-1](https://github.com/AlphanAksoyoglu/AI-jsbach-music-generator/blob/main/images/presentation/slide1.png?raw=true)
![Slide-2](https://github.com/AlphanAksoyoglu/AI-jsbach-music-generator/blob/main/images/presentation/slide2.png?raw=true)
![Slide-3](https://github.com/AlphanAksoyoglu/AI-jsbach-music-generator/blob/main/images/presentation/slide3.png?raw=true)
![Slide-4](https://github.com/AlphanAksoyoglu/AI-jsbach-music-generator/blob/main/images/presentation/slide4.png?raw=true)
![Slide-5](https://github.com/AlphanAksoyoglu/AI-jsbach-music-generator/blob/main/images/presentation/slide5.png?raw=true)
![Slide-6](https://github.com/AlphanAksoyoglu/AI-jsbach-music-generator/blob/main/images/presentation/slide6.png?raw=true)
![Slide-7](https://github.com/AlphanAksoyoglu/AI-jsbach-music-generator/blob/main/images/presentation/slide7.png?raw=true)
![Slide-8](https://github.com/AlphanAksoyoglu/AI-jsbach-music-generator/blob/main/images/presentation/slide8.png?raw=true)
![Slide-9](https://github.com/AlphanAksoyoglu/AI-jsbach-music-generator/blob/main/images/presentation/slide9.png?raw=true)
![Slide-10](https://github.com/AlphanAksoyoglu/AI-jsbach-music-generator/blob/main/images/presentation/slide10.png?raw=true)