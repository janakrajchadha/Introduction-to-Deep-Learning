#Classifying the sentiment of IMDB Movie reviews using Recurrent Neural Networks
#Deep learning library used - TFLearn

from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

#Loading the dataset
train, test, _ = imdb.load_data(path='imdb.pkl', n_words = 15000, valid_portion = 0.1)

train_features, train_labels = train
test_features, test_labels = test

#Pre processing the data
#Sequence padding

train_features = pad_sequences(train_features , maxlen = 100, value = 0.)
test_features = pad_sequences(test_features , maxlen = 100, value = 0.)

#converting the labels as well
train_labels = to_categorical(train_labels, nb_classes = 2)
test_labels = to_categorical(test_labels, nb_classes = 2)

#Building the main network using tflearn

net1 = tflearn.input_data([None,100])
net1 = tflearn.embedding(net1, input_dim = 15000, output_dim = 128)
net1 = tflearn.lstm(net1, 128, dropout = 0.8)
net1 = tflearn.fully_connected(net1, 2, activation = 'softmax')
net1 = tflearn.regression(net1, optimizer = 'adam', learning_rate = 0.00005, loss = 'categorical_crossentropy')

#Training time

model = tflearn.DNN(net1, tensorboard_verbose = 0)
model.fit(train_features, train_labels, validation_set = (test_features, test_labels), show_metric = True, batch_size = 32)
