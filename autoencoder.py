import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import preprocessing
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Plot a confusion matrix.
# cm is the confusion matrix, names are the names of the classes.
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

# Plot an ROC. pred - the predictions, y - the expected output.
def plot_roc(pred,y):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

#Autoencoder
filename = 'preprocessed_data.csv'
df = pd.read_csv(filename)
df = df.astype("str")
df.head()
df.info()
df.columns=["v1"]

X_train = df.v1

max_words = 1500
max_len = 500
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

code_dim = 250

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

# generate reduced data by using "encoders"
training_data_reduced = encoder.predict(sequences_matrix)

#Model for Manually Annotated Content - classified as High or Low
filename2 = 'manually_annotated.csv'
df = pd.read_csv(filename2)
df = df.astype('str')
df.head()
df.info()
X = df.v2
Y = df.v1
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.15)
max_words = 3000
max_len = 500
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(x_train)
sequences = tok.texts_to_sequences(x_train)
sequences_matrix_m = sequence.pad_sequences(sequences,maxlen=max_len)

model = Sequential()
model.add(auto_encoder.layers[1])
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(sequences_matrix_m, y_train, verbose=1, epochs=10, validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

#Measure accuracy

test_sequences = tok.texts_to_sequences(x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(test_sequences_matrix, y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

#pred = model.predict(x_test) 
pred = model.predict(test_sequences_matrix) #<-y_test
#pred = np.argmax(pred, axis=1)
y_compare = np.argmax(pred, axis=1)
#score = metrics.accuracy_score(y_compare, pred)
#print("Final accuracy: {}".format(score))#Create NN <- Fully Connected layers

import numpy as np

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(y_compare, pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm, pred)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, pred, title='Normalized confusion matrix')

plt.show()

pred = model.predict(test_sequences_matrix)
#pred = pred[:,1] # Only positive cases
plot_roc(pred,y_compare)







