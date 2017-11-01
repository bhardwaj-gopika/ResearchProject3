import numpy as np

#Sample Data
documents = ["IRS Commissioner John Koskinen testifies before the Senate Finance Committee about the 100,000 taxpayers at risk for identity theft after hackers breached a tool for college students. A financial aid tool for college students helped hackers steal up to +ACQ-30 million from the US government. Nearly 100,000 people are at risk for identity theft after hackers breached the IRS's Data Retrieval Tool, which parents use to transfer financial information for their kids using the Free Application for Federal Student Aid. In 2015, 17 million students used FAFSA to file for financial aid. Fraudulent tax returns have become a growing issue for the IRS, as hackers find more sophisticated measures to steal financial documents online. The agency lost +ACQ-5.8 billion in 2013 alone from sending tax refunds to thieves filing in other people's names. These schemes have targeted schools, hospitals and restaurants, and college students are the latest victims.  IRS Commissioner John Koskinen testified to the Senate Finance Committee on Thursday, revealing thousands of people could be hit by identity theft from the breach. The agency delayed refunds from going out to 52,000 taxpayers until they can verify they're real requests. +ACIAIg-It was clear that some of that activity was legitimate students, some of it was criminals,+ACIAIg- Koskinen said. +ACIAIg-So we shut the system down.+ACIAIg- The tool, which allowed applicants to automatically upload their tax information, also allowed hackers to pose as 8,000 college students in tax refund requests. They would start the financial aid process like a normal student, and then use the IRS tool to automatically populate tax information for the student and parents.  Using that stolen tax information, identity thieves filed fraudulent tax returns, stealing +ACQ-30 million from the IRS. Up to 14,000 other phony tax refunds were blocked from the IRS. The Department of Education and the IRS disabled the tool in March, during a critical time when students are applying for loans, and said it wouldn't return online until the fall. The IRS first learned about the breach in September 2016, but delayed shutting down the tool then because millions of students depend on it.  +ACIAIg-As soon as there was any indication of criminal activity, we would have to take that application down,+ACIAIg- Koskinen said. +ACIAIg-That occurred, as we monitored, in through the early part of Feburary.+ACIAIg- Students can still fill out their application manually without the tool, but the process takes a longer time. The IRS has notified 100,000 people that their information is at risk.   CNET Magazine: Check out a sample of the stories in CNET's newsstand edition. Life, disrupted: In Europe, millions of refugees are still searching for a safe place to settle. Tech should be part of the solution. But is it?"]

sentences = [[word for word in document.lower().split()] for document in documents] #<type 'list'>

# Sample code to prepare word2vec word embeddings    

import gensim
from gensim.models import word2vec

word_model = gensim.models.word2vec.Word2Vec(sentences, size=1, min_count = 1, window = 50)#<class 'gensim.models.word2vec.Word2Vec'>
word_vectors = word_model.wv #<class 'gensim.models.keyedvectors.KeyedVectors'>
#data = np.array(word_vectors, ndmin = 2, dtype = object) #<type 'numpy.ndarray'>
labels = np.array([0.214285714286], ndmin = 2 , dtype = object) #<type 'numpy.ndarray'> Normalised Label

vector_dim = 1
# convert the wv word vectors into a numpy matrix that is suitable for insertion
# into our TensorFlow and Keras models
embedding_matrix = np.zeros((len(word_model.wv.vocab), vector_dim))
for i in range(len(word_model.wv.vocab)):
    embedding_vector = word_model.wv[word_model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

data = np.squeeze(np.asarray(embedding_matrix))
data = np.array(data, ndmin = 2, dtype = object)
# create the model
print(data)
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

embedding_layer = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix])
#embedding_layer = Embedding(input_dim=word_model.wv.syn0.shape[0], output_dim=word_model.wv.syn0.shape[1], weights=[word_model.wv.syn0])
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(embedding_matrix.shape[1], return_sequences=True))
model.add(LSTM(embedding_matrix.shape[1], return_sequences=False))
model.add(Dense(embedding_matrix.shape[0]))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(data, labels, epochs=10, batch_size=32)

# Evaluate the network
loss, accuracy = model.evaluate(data, labels)
print("Loss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100)) #ALSO PRINT IN OUTPUT CSV

import numpy as numpy 
# Make predictions
probabilities = model.predict(data)
predictions = [x for x in probabilities]
accuracy = numpy.mean(predictions == labels)
pred = accuracy*100 #Prediction Accuracy
print(pred)


