__author__ = 'HyNguyen'

import pickle
from nnlayers import FullConectedLayer, SoftmaxLayer, MyConvLayer, ProjectionLayer
from preparedata4cnn import Vocabulary, gen_data_from_sentence_indexs
import numpy as np
import theano
import theano.tensor as T
import time

class MySentence(object):
    def __init__(self, sentence, syllables ,  words_index, labels=[] ):
        self.sentence = sentence
        self.syllables = syllables
        self.words_index = words_index
        self.labels = labels
        self.sentence_ws = ""
        self.data_for_cnn = None

    @classmethod
    def mysentence_from_string(cls, string, vocabulary):
        indexs, syllables, labels = vocabulary.sen_2_index(string)
        return MySentence(string, syllables, indexs)

    def gen_data_for_cnn(self):
        X = np.array([[0,0,0,0,0]])
        Y = np.array([0])
        xx, yy = gen_data_from_sentence_indexs(self.words_index, self.labels)
        X = np.concatenate((X,xx), axis=0)
        Y = np.concatenate((Y,yy))
        return X, Y

    def gen_result(self, y):
        text_result = ""
        for i in range(1,y.shape[0],1):
            if y[i] == 1:
                text_result = text_result + self.syllables[i-1] + "_"
            else:
                text_result = text_result + self.syllables[i-1] + " "
        return (text_result)

if __name__ == "__main__":

    start_preparedata = time.clock()
    vocabulary = Vocabulary.load("model/vocab.bin")
    mysentences = []
    filename = "/Users/HyNguyen/Documents/Research/Data/VCL/vcl.1000.txt"
    with open(filename) as f:
        lines = f.readlines()

    total_number = 0

    X_test, Y_test = None, None
    for i,line in enumerate(lines):
        mysentence = MySentence.mysentence_from_string(line, vocabulary)
        total_number += len(mysentence.syllables)
        mysentences.append(mysentence)
        xx, yy = mysentence.gen_data_for_cnn()
        mysentence.data_for_cnn = xx
        if X_test is None and Y_test is None:
            X_test, Y_test = xx, yy
        else:
            X_test = np.concatenate((X_test,xx))
            Y_test = np.concatenate((Y_test,yy))
    end_preparedata = time.clock()
    print("Time prepare data = ", end_preparedata - start_preparedata, " s")

    start_loadmodel = time.clock()
    with open("model/saveweight2.bin", mode="rb") as f:
        params = pickle.load(f)
    end_loadmodel = time.clock()
    print("Time for load NN model = ", end_loadmodel - start_loadmodel, " s")

    start_confignn = time.clock()
    vocab_size = len(vocabulary.word_index.keys())
    batch_size= 10000
    number_featuremaps = 500
    sentence_length = 5
    embsize = 500
    learning_rate = 0.1
    filter_shape_encode = (number_featuremaps,1,3,embsize)
    rng = np.random.RandomState(23455)

    X = T.matrix('X')

    image_shape = (batch_size,1,sentence_length,embsize)

    layer_projection = ProjectionLayer(X, vocab_size, embsize, (batch_size, sentence_length), [params[0]])

    layer_conv = MyConvLayer(rng, layer_projection.output,image_shape=(batch_size,1,sentence_length,embsize),filter_shape=filter_shape_encode,border_mode="valid",activation = T.tanh, params=params[1:3])

    layer_input = layer_conv.output.flatten(2)
    layer_input_shape = (batch_size,layer_conv.output_shape[1] * layer_conv.output_shape[2] * layer_conv.output_shape[3])
    layer_hidden = FullConectedLayer(layer_input, layer_input_shape[1] , 100, activation = T.tanh, params=params[3:5])

    layer_classification =  SoftmaxLayer(input=layer_hidden.output, n_in=100, n_out=2, params=params[5:7])

    pred = layer_classification.y_pred

    pred_function = theano.function(inputs=[X], outputs=pred, on_unused_input="ignore")
    show_function = theano.function(inputs=[X], outputs=layer_conv.conv_out, on_unused_input="ignore")
    end_confignn = time.clock()
    print("Time for config network ", end_confignn - start_confignn, " s")

    Y_pred = None
    start_pred = time.clock()
    batch_number = int(X_test.shape[0] / batch_size)
    for batch_count in range(batch_number+1):
        if batch_count*batch_size + batch_size > X_test.shape[0]:
            X_mnb = X_test[batch_count*batch_size:]
            X_mnb = np.concatenate((X_mnb, np.zeros((batch_size - X_mnb.shape[0],sentence_length))))
        else:
            X_mnb = X_test[batch_count*batch_size:batch_count*batch_size + batch_size]
        Y_mnb = pred_function(X_mnb)
        if Y_pred is None:
            Y_pred = Y_mnb
        else:
            Y_pred = np.concatenate((Y_pred,Y_mnb))

    end_pred = time.clock()
    print("Time for predict label ", end_pred - start_pred, " s")

    print("input_shape ", X_test.shape)
    print("predict_shape ",Y_pred.shape)

    start_write = time.clock()
    fo = open(filename +".ws.txt", mode="w")
    old_index = 0
    for mysentence in mysentences:
        length = mysentence.data_for_cnn.shape[0]
        mysentence.labels = Y_pred[old_index:old_index+length]
        old_index += length
        fo.write(mysentence.gen_result(mysentence.labels)+"\n")
    fo.close()
    end_write = time.clock()
    print("Time write to file = ", end_write - start_write, " s")
    print("total number: ",total_number)


    # B1: change to words_index and save sentence
    # B2: