'''
Runs the model
The model consists of 3 bidirectional LSTM layers of size 256
And a dense layer that matches the size of the output categories

Due to the sheer size of the input data ~64GB even with uint8 sparse matrices
we use a batch generator for input and validation

Please see README for more details...

Important ToDo:
The size of the Dense layer is currently a manual input this should be adjusted
'''

from data_utils import DataIOHandler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Bidirectional, BatchNormalization, Activation, Dense, Input, Dropout
import pydot
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint

def batch_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0

    index = np.arange(samples_per_epoch)
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]

        X_batch = np.array([i.todense() for i in X_data[index_batch,]])
        sopr = y_data[0][index_batch]
        alto = y_data[1][index_batch]
        teno = y_data[2][index_batch]
        bass = y_data[3][index_batch]
        bas2 = y_data[4][index_batch]
        keys = y_data[5][index_batch]
        star = y_data[6][index_batch]
        ends = y_data[7][index_batch]
        counter += 1
        yield ({"input_num": X_batch},
        {
        'sopr_out':sopr,
        'alto_out':alto,
        'teno_out':teno,
        'bass_out':bass,
        'bas2_out':bas2,
        'keys_out':keys,
        'star_out':star,
        'ends_out':ends
        })
        if (counter >= number_of_batches):
            counter=0

def validation_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0

    index = np.arange(samples_per_epoch)
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]

        X_batch = np.array([i.todense() for i in X_data[index_batch,]])
        sopr = y_data[0][index_batch]
        alto = y_data[1][index_batch]
        teno = y_data[2][index_batch]
        bass = y_data[3][index_batch]
        bas2 = y_data[4][index_batch]
        keys = y_data[5][index_batch]
        star = y_data[6][index_batch]
        ends = y_data[7][index_batch]
        counter += 1
        yield ({"input_num": X_batch},
        {
        'sopr_out':sopr,
        'alto_out':alto,
        'teno_out':teno,
        'bass_out':bass,
        'bas2_out':bas2,
        'keys_out':keys,
        'star_out':star,
        'ends_out':ends
        })
        if (counter >= number_of_batches):
            counter=0

if tf.test.gpu_device_name():
    print('Running TF On Default GPU Device: {}\n'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF\n")

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('Memory Growth Mode Set\n')
except:
      # Invalid device or cannot modify virtual devices once initialized.
    pass

data_handler = DataIOHandler()

outlist = data_handler.get_pickle_data('p5t4_outs.pickle')

sopr_outs = outlist[0]
alto_outs = outlist[1]
teno_outs = outlist[2]
bass_outs = outlist[3]
bas2_outs = outlist[4]
keys_outs = outlist[5]
star_outs = outlist[6]
ends_outs = outlist[7]

# print('Getting outputs\n')
# sopr_outs = get_pickle_data('./data/p5t4_onehot_sopr_out.pickle')
# alto_outs = get_pickle_data('./data/p5t4_onehot_alto_out.pickle')
# teno_outs = get_pickle_data('./data/p5t4_onehot_teno_out.pickle')
# bass_outs = get_pickle_data('./data/p5t4_onehot_bass_out.pickle')
# bas2_outs = get_pickle_data('./data/p5t4_onehot_bas2_out.pickle')
# keys_outs = get_pickle_data('./data/p5t4_onehot_keys_out.pickle')
# star_outs = get_pickle_data('./data/p5t4_onehot_star_out.pickle')
# ends_outs = get_pickle_data('./data/p5t4_onehot_ends_out.pickle')
# outlist = [sopr_outs, alto_outs, teno_outs, bass_outs, bas2_outs, keys_outs, star_outs, ends_outs]

# print('Loaded Outputs\n')

print('Getting inputs, might take a bit\n')
inputs_data = data_handler.get_pickle_data('p5t4_ins.pickle')
print('Loaded Inputs\n')

print('Setting Parameters\n')
full_data_size = inputs_data.shape[0]
batch_s = 64
data_size = int((full_data_size//(batch_s*10))*(batch_s*10))
train_size = int(data_size*0.8)
validation_size = int(data_size *0.2)

train_inputs = inputs_data[0:train_size]
train_outputs = [i[0:train_size] for i in outlist]
val_inputs = inputs_data[train_size:train_size+validation_size]
val_outputs = [i[train_size:train_size+validation_size] for i in outlist]
print(f'''Will train the model with:\n
        Data Size of:{data_size}\n
        Train Size:{train_size}\n
        Validation Size:{validation_size}\n
        Batch Size:{batch_s}\n
        ''')

input_num = Input(batch_size=batch_s,shape=(512,605),name='input_numeric')
lstm1 = Bidirectional(LSTM(256,stateful=True,return_sequences=True))(input_num)
lstm1_drop = Dropout(0.3)(lstm1)
lstm2 = Bidirectional(LSTM(256,stateful=True,return_sequences=True))(lstm1_drop)

lstm2_drop = Dropout(0.3)(lstm2)
lstm3 = Bidirectional(LSTM(256,stateful=True))(lstm2_drop)
lstm3_drop = Dropout(0.3)(lstm3)
dense1 = Dense(387, activation='relu')(lstm3)
dense1_drop = Dropout(0.3)(dense1)


sopr_out = Dense(sopr_outs.shape[1], activation='softmax',name='sopr_out')(dense1_drop)
alto_out = Dense(alto_outs.shape[1], activation='softmax',name='alto_out')(dense1_drop)
teno_out = Dense(teno_outs.shape[1], activation='softmax',name='teno_out')(dense1_drop)
bass_out = Dense(bass_outs.shape[1], activation='softmax',name='bass_out')(dense1_drop)
bas2_out = Dense(bas2_outs.shape[1], activation='softmax',name='bas2_out')(dense1_drop)
keys_out = Dense(keys_outs.shape[1], activation='softmax',name='keys_out')(dense1_drop)
star_out = Dense(star_outs.shape[1], activation='softmax',name='star_out')(dense1_drop)
ends_out = Dense(ends_outs.shape[1], activation='softmax',name='ends_out')(dense1_drop)


model = keras.Model(
    inputs = input_num,
    outputs = [sopr_out,alto_out,teno_out,bass_out,bas2_out,keys_out,star_out,ends_out]
)

model.compile(
    optimizer='adam',
    metrics=['accuracy'],
    loss={
        'sopr_out':'categorical_crossentropy',
        'alto_out':'categorical_crossentropy',
        'teno_out':'categorical_crossentropy',
        'bass_out':'categorical_crossentropy',
        'bas2_out':'categorical_crossentropy',
        'keys_out':'categorical_crossentropy',
        'star_out':'categorical_crossentropy',
        'ends_out':'categorical_crossentropy'
    }
)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-VAL_LOSS-{val_loss:.2f}.h5',
                                                 save_weights_only=False,
                                                 monitor='val_loss',
                                                 save_best_only=False,
                                                 verbose=0)

history = model.fit_generator(
    generator=batch_generator(train_inputs, train_outputs, batch_s),
    validation_data=validation_generator(val_inputs, val_outputs, batch_s),
    epochs=2,
    steps_per_epoch=train_size/batch_s,
    validation_steps=validation_size/batch_s,
    shuffle=False,
    callbacks=[cp_callback]
)
