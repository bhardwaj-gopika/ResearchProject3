import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import csv

filename = 'data6.csv'
df = pd.read_csv(filename, error_bad_lines=False, quoting=csv.QUOTE_NONE)
df.head()
df.info()
df.columns=["v1"]

X_train = df.v1
#y_data = df.target



max_words = 1500
max_len = 250
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

code_dim = 16

def auto_encoder_model():   
    inputs = Input(shape = [max_len], name = 'inputs')	                         # input layer
    layer = Embedding(max_words,50,input_length=max_len)(inputs)	
    code = Dense(code_dim, activation = 'relu', name = 'code')(inputs)                  # hidden layer => represents "codes"
    outputs = Dense(max_len, activation = 'softmax', name = 'output')(code)    # output layer

    auto_encoder = Model(inputs = inputs, outputs = outputs)

    encoder = Model(inputs = inputs, outputs = code)

    decoder_input = Input(shape = (code_dim,))
    decoder_output = auto_encoder.layers[-1]
    decoder = Model(inputs = decoder_input, outputs = decoder_output(decoder_input))

    auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto_encoder

encoder, decoder, auto_encoder = auto_encoder_model()
auto_encoder.fit(sequences_matrix, sequences_matrix, epochs = 10, batch_size = 50, verbose = 0)
print(auto_encoder.summary())
auto_encoder.save('autoencoder.h5')
print(encoder.summary())
encoder.save('encoder.h5')
print(auto_encoder.layers)
# generate reduced data by using "encoders"
training_data_reduced = encoder.predict(sequences_matrix)
#test_data_reduced = encoder.predict(X_test)


print(training_data_reduced.shape)
#print(test_data_reduced.shape)

print(training_data_reduced[0])    # first insance of reduced training data
#print(test_data_reduced[0])        # first instance of reduced test data





