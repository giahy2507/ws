__author__ = 'HyNguyen'

import pickle

if __name__ == "__main__":

    with open("data/vtb.pre.txt.train.np", mode="rb") as f:
        X_train, Y_train, X_validation, Y_validation = pickle.load(f)

    a = 2.71170000e+04
    print (int(a))
    print(X_train.shape, Y_train.shape,X_validation.shape,Y_validation.shape)