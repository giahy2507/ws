__author__ = 'HyNguyen'

from nnlayers import FullConectedLayer, SoftmaxLayer, MyConvLayer, ProjectionLayer
import pickle
import numpy as np
import theano
import theano.tensor as T
from preparedata4cnn import Vocabulary
import sys
from lasagne.updates import adam
import os
import time

if __name__ == "__main__":

    with open("data/vtb.pre.txt.train.np", mode="rb") as f:
        X_train, Y_train, X_validation, Y_validation = pickle.load(f)

    X_train_size = int(X_train.shape[0]/10)
    X_train = np.array(X_train[:X_train_size],dtype= np.int32)
    Y_train = np.array(Y_train[:X_train_size],dtype= np.int32)

    x_test_size = int(0.1 * X_validation.shape[0])

    X_test = np.array(X_validation[:x_test_size], dtype=np.int32)
    Y_test = np.array(Y_validation[:x_test_size], dtype=np.int32)

    X_validation = np.array(X_validation[:x_test_size], dtype=np.int32)
    Y_validation = np.array(Y_validation[:x_test_size],dtype= np.int32)

    with open("model/vocab.bin", mode="rb") as f:
        vocabulary = pickle.load(f)

    vocab_size = len(vocabulary.word_index.keys())
    sys.stdout.write(str(X_train.shape)+ str(Y_train.shape) + str(X_validation.shape) + str(Y_validation.shape) + str(X_test.shape) + str(Y_test.shape))
    sys.stdout.flush()

    batch_size=100
    number_featuremaps = 500
    sentence_length = 5
    embsize = 500
    learning_rate = 0.1
    filter_shape_encode = (number_featuremaps,1,3,embsize)
    rng = np.random.RandomState(23455)

    X = T.matrix('X')
    Y = T.ivector('Y')
    image_shape = (batch_size,1,sentence_length,embsize)

    params = [None] * 7
    if os.path.isfile("model/saveweight.bin"):
        with open("model/saveweight.bin", mode="rb") as f:
            params = pickle.load(f)

    layer_projection = ProjectionLayer(X, vocab_size, embsize, (batch_size, sentence_length), [params[0]])

    layer_conv = MyConvLayer(rng, layer_projection.output,image_shape=(batch_size,1,sentence_length,embsize),filter_shape=filter_shape_encode,border_mode="valid",activation = T.tanh, params=params[1:3])

    layer_input = layer_conv.output.flatten(2)
    layer_input_shape = (batch_size,layer_conv.output_shape[1] * layer_conv.output_shape[2] * layer_conv.output_shape[3])
    layer_hidden = FullConectedLayer(layer_input, layer_input_shape[1] , 100, activation = T.tanh, params=params[3:5])

    layer_classification =  SoftmaxLayer(input=layer_hidden.output, n_in=100, n_out=2, params=params[5:7])

    err = layer_classification.error(Y)

    cost = layer_classification.negative_log_likelihood(Y) + 0.001*(layer_projection.L2 + layer_conv.L2 +layer_classification.L2 + layer_hidden.L2)

    params = layer_projection.params + layer_conv.params + layer_hidden.params + layer_classification.params
    
    updates = adam(cost,params)
    
    """
    gparams = []
    for param in params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)
    updates = []
    for param, gparam in zip(params, gparams):
        updates.append((param, param - learning_rate* gparam))
    """

    train_model = theano.function(inputs=[X,Y], outputs=[cost, err],updates=updates,on_unused_input="ignore",allow_input_downcast=True)
    valid_model = theano.function(inputs=[X,Y], outputs=[cost, err],on_unused_input="ignore",allow_input_downcast=True)
    show_function = theano.function(inputs=[X,Y], outputs=err, on_unused_input="ignore",allow_input_downcast=True)

    counter = 0
    best_valid_err = 100
    early_stop = 50
    epoch_i = 0

    train_rand_idxs = list(range(0,X_train.shape[0]))
    valid_rand_idxs = list(range(0,X_validation.shape[0]))
    test_rand_idxs = list(range(0,X_test.shape[0]))

    while counter < early_stop:
        epoch_i +=1

        train_costs = []
        train_errs = []

        valid_costs = []
        valid_errs = []

        test_costs = []
        test_errs = []

        np.random.shuffle(train_rand_idxs)
        batch_number = int(X_train.shape[0] / batch_size)
        start_train = time.clock()
        for batch_i in range(batch_number):
            mnb_X = X_train[train_rand_idxs[batch_i*batch_size: batch_i*batch_size + batch_size]]
            mnb_Y  = Y_train[train_rand_idxs[batch_i*batch_size: batch_i*batch_size + batch_size]]
            if mnb_X.shape[0] != batch_size:
                continue
            train_cost, train_err = train_model(mnb_X, mnb_Y)
            train_costs.append(train_cost)
            train_errs.append(train_err)
        end_train = time.clock()
        np.random.shuffle(valid_rand_idxs)
        batch_number = int(X_validation.shape[0] / batch_size)
        for start_idx in range(batch_number):
            mnb_X = X_validation[valid_rand_idxs[batch_i*batch_size: batch_i*batch_size + batch_size]]
            mnb_Y  = Y_validation[valid_rand_idxs[batch_i*batch_size: batch_i*batch_size + batch_size]]
            if mnb_X.shape[0] != batch_size:
                continue
            valid_cost, valid_err = valid_model(mnb_X, mnb_Y)
            valid_costs.append(valid_cost)
            valid_errs.append(valid_err)

        np.random.shuffle(test_rand_idxs)
        batch_number = int(X_test.shape[0] / batch_size)
        for start_idx in range(batch_number):
            mnb_X = X_test[test_rand_idxs[batch_i*batch_size: batch_i*batch_size + batch_size]]
            mnb_Y  = Y_test[test_rand_idxs[batch_i*batch_size: batch_i*batch_size + batch_size]]
            if mnb_X.shape[0] != batch_size:
                continue
            tess_cost, test_err = valid_model(mnb_X, mnb_Y)
            test_costs.append(tess_cost)
            test_errs.append(test_err)

        train_err = np.mean(np.array(train_errs))
        train_cost = np.mean(np.array(train_costs))
        val_err = np.mean(np.array(valid_errs))
        val_cost = np.mean(np.array(valid_costs))
        test_err = np.mean(np.array(test_errs))
        test_cost = np.mean(np.array(test_costs))

        if best_valid_err > val_err:
            best_valid_err = val_err
            sys.stdout.write("Epoch "+str(epoch_i)+" Train cost: "+ str(train_cost)+ "Train err: "+ str(train_err) + "Time: " + end_train - start_train + " Validation cost: "+ str(valid_cost)+" Validation err "+ str(val_err) + " Test cost: "+ str(test_cost)+" Test err "+ str(test_err)  + ",counter "+str(counter)+ " __best__ \n")
            sys.stdout.flush()
            counter = 0
            with open("model/saveweight.bin", mode="wb") as f:
                pickle.dump(params,f)
        else:
            counter +=1
            sys.stdout.write("Epoch "+str(epoch_i)+" Train cost: "+ str(train_cost)+ "Train err: "+ str(train_err) + "Time: " + end_train - start_train + " Validation cost: "+ str(valid_cost)+" Validation err "+ str(val_err) + " Test cost: "+ str(test_cost)+" Test err "+ str(test_err)  + ",counter "+str(counter)+ "\n")
            sys.stdout.flush()
