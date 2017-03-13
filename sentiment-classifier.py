#Classifying the sentiment of IMDB Movie reviews using Recurrent Neural Networks
#Deep learning library used - TFLearn

from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

#Loading the dataset
train, test, _ = imdb.load_data(path='imdb.pkl', n_words = 15000, valid_portion = 0.15)

trainA, trainB = train
testA, testB = test

#Pre processing the data
#Sequence padding 

trainA = pad_sequences(trainA , maxlen = 100, value = 0.)
testA = pad_sequences(testA , maxlen = 100, value = 0.)

#converting the labels as well
trainB = to_categorical(trainB, nb_classes = 2)
testB = to_categorical(testB, nb_classes = 2)

#Building the main network using tflearn

net1 = tflearn.input_data([None,100])
net1 = tflearn.embedding(net1, input_dim = 15000, output_dim = 128)
net1 = tflearn.lstm(net1, 128, dropout = 0.8)
net1 = tflearn.fully_connected(net1, 2, activation = 'softmax')
net1 = tflearn.regression(net1, optimizer = 'adam', learning_rate = 0.00005, loss = 'categorical_crossentropy')

#Training time

model = tflearn.DNN(net1, tensorboard_verbose = 0)
model.fit(trainA, trainB, validation_set = (testA, testB), show_metric = True, batch_size = 32)
