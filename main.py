__author__ = 'HyNguyen'

import pickle
from preparedata4cnn import Vocabulary

if __name__ == "__main__":

    with open("data/vtb.pre.txt.train.nparray", mode="rb") as f:
        X_train, Y_train, X_validation, Y_validation = pickle.load(f)

    print(X_train.shape, Y_train.shape,X_validation.shape,Y_validation.shape)

    with open("model/vocab.bin", mode="rb") as f:
        vocabulary = pickle.load(f)

    index = vocabulary.get_index("#number")
    print(index)