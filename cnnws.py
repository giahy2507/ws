__author__ = 'HyNguyen'

from nnlayers import FullConectedLayer, SoftmaxLayer, MyConvLayer, ProjectionLayer
import pickle
import numpy as np
import theano
import theano.tensor as T
from preparedata4cnn import Vocabulary

if __name__ == "__main__":
    print("__main__")

    with open("data/vtb.pre.txt.train.np", mode="rb") as f:
        X_train, Y_train, X_validation, Y_validation = pickle.load(f)

    X_train = np.array(X_train,dtype= np.int32)
    Y_train = np.array(Y_train,dtype= np.int32)
    X_test, Y_test = X_train[:100], Y_train[:100]

    with open("model/vocab.bin", mode="rb") as f:
        vocabulary = pickle.load(f)

    vocab_size = len(vocabulary.word_index.keys())

    print(X_train.shape, Y_train.shape,X_validation.shape,Y_validation.shape,X_test.shape,Y_test.shape)

    batch_size=100
    number_featuremaps = 20
    sentence_length = 5
    embsize = 500
    learning_rate = 0.1
    filter_shape_encode = (20,1,3,embsize)
    rng = np.random.RandomState(23455)

    X = T.matrix('X')
    Y = T.ivector('Y')
    image_shape = (batch_size,1,sentence_length,embsize)

    layer_projection = ProjectionLayer(X,vocab_size,embsize,X.shape,[None])

    layer_conv = MyConvLayer(rng, layer_projection.output,image_shape=(batch_size,1,sentence_length,embsize),filter_shape=filter_shape_encode,border_mode="valid",activation = T.nnet.sigmoid, params=[None,None])

    layer_input = layer_conv.output.flatten(2)
    layer_input_shape = (batch_size,layer_conv.output_shape[1] * layer_conv.output_shape[2] * layer_conv.output_shape[3])
    layer_hidden = FullConectedLayer(layer_input, layer_input_shape[1] , 100, activation = T.nnet.sigmoid, params=[None,None])

    layer_classification =  SoftmaxLayer(input=layer_hidden.output, n_in=100, n_out=2)

    err = layer_classification.error(Y)

    cost = layer_classification.negative_log_likelihood(Y) + 0.001*(layer_projection.L2 + layer_conv.L2 +layer_classification.L2 + layer_hidden.L2)

    params = layer_projection.params + layer_conv.params + layer_hidden.params + layer_classification.params

    gparams = []
    for param in params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)
    updates = []
    for param, gparam in zip(params, gparams):
        updates.append((param, param - learning_rate* gparam))

    train_model = theano.function(inputs=[X,Y], outputs=[cost, err],updates=updates,on_unused_input="ignore")
    valid_model = theano.function(inputs=[X,Y], outputs=[cost, err],on_unused_input="ignore")
    show_function = theano.function(inputs=[X,Y], outputs=[layer_projection.output, layer_conv.output, layer_hidden.input, layer_classification.p_y_given_x, err], on_unused_input="ignore")

    counter = 0
    best_valid_err = 100
    early_stop = 50
    epoch_i = 0

    train_rand_idxs = list(range(0,X_train.shape[0]))
    valid_rand_idx = list(range(0,X_validation.shape[0]))
    # print(train_rand_idxs, valid_rand_idx)
    while counter < early_stop:
        epoch_i +=1

        train_costs = []
        train_errs = []

        valid_costs = []
        valid_errs = []

        np.random.shuffle(train_rand_idxs)
        for start_idx in range(0, X_train.shape[0], batch_size):
            if start_idx + batch_size > X_train.shape[0]:
                # print("Out of size")
                continue
            mnb_X = X_train[train_rand_idxs[start_idx: start_idx + batch_size]]
            mnb_Y  = Y_train[train_rand_idxs[start_idx: start_idx + batch_size]]
            train_cost, train_err = train_model(mnb_X, mnb_Y)
            train_costs.append(train_cost)
            train_errs.append(train_err)
            print ("Epoch "+str(epoch_i)+" Train cost: "+ str(train_cost)+ "Train mae: "+ str(train_err))

        np.random.shuffle(valid_rand_idx)
        for start_idx in range(0, X_validation.shape[0], batch_size):
            if start_idx + batch_size > X_validation.shape[0]:
                # print("Out of size")
                continue
            mnb_X = X_validation[valid_rand_idx[start_idx: start_idx + batch_size]]
            mnb_Y  = Y_validation[valid_rand_idx[start_idx: start_idx + batch_size]]
            valid_cost, valid_err = valid_model(mnb_X, mnb_Y)
            valid_costs.append(valid_cost)
            valid_errs.append(valid_err)

        train_err = np.mean(np.array(train_errs))
        train_cost = np.mean(np.array(train_costs))
        val_err = np.mean(np.array(valid_errs))
        val_cost = np.mean(np.array(valid_costs))

        if best_valid_err > val_err:
            best_valid_err = val_err
            print ("Epoch "+str(epoch_i)+" Train cost: "+ str(train_cost)+ "Train mae: "+ str(train_err) + " Validation cost: "+ str(valid_cost)+" Validation mae "+ str(val_err)  + ",counter "+str(counter)+ " __best__ ")
            counter = 0
            with open("model/saveweight.bin", mode="wb") as f:
                pickle.dump(params,f)
        else:
            counter +=1
            print ("Epoch "+str(epoch_i)+" Train cost: "+ str(train_cost)+ "Train mae: "+ str(train_err) + " Validation cost: "+ str(valid_cost)+" Validation mae "+ str(val_err)  + ",counter "+str(counter))