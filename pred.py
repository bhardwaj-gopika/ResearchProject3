import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping 
from keras.callbacks import ModelCheckpoint
import csv
import matplotlib.pyplot as plt

#X_test
filename = 'Preprocessed_data.csv'
df = pd.read_csv(filename, error_bad_lines=False, quoting=csv.QUOTE_NONE)
df.head()
df.info()
df.columns=["v1"]
x = df.v1
max_words = 3000
max_len = 500
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(x)
sequences = tok.texts_to_sequences(x)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

code_dim = 250

#Autoencoder Function
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
auto_encoder.fit(sequences_matrix, sequences_matrix, epochs = 10, batch_size = 50, verbose = 1)
# generate reduced data by using "encoders"
training_data_reduced = encoder.predict(sequences_matrix) #<-x_test
x_test = training_data_reduced

#X_train, Y_train
filename2 = 'manually_annotated.csv'
df = pd.read_csv(filename2 ,delimiter=',',encoding='latin-1')
df.head()
df.info()
X_train = df.v2
Y_train = df.v1
le = LabelEncoder()
Y_train = le.fit_transform(Y)
Y_train = Y_train.reshape(-1,1)
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)
max_words = 1500
max_len = 250
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

model = Sequential()
model.add(Dense(20, input_dim=x.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(Y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath="best_weights.hd5", verbose=0, save_best_only=True) #save best model
model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[monitor, checkpointer], verbose=0, epochs=10)
model.load_weights('best_weights.hd5') #load weights from best model

#Measure accuracy
pred = model.predict_classes(x_test) #<-y_test 
pred = np.argmax(pred, axis=1)
y_compare = np.argmax(y_test, axis=1)
score = metrics.accuracy_score(y_compare, pred)
print("Final accuracy: {}".format(score))#Create NN <- Fully Connected layers


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

#Confusion Matrix
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Compute confusion matrix
cm = confusion_matrix(y_compare, pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm, diagnosis)
plt.show()

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


#ROC curve
pred = model.predict(x_test)
pred = pred[:,1] # Only positive cases
plot_roc(pred,y_compare)
