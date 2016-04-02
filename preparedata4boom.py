import os
import pickle
import time
import sys

class Word(object):
    def __init__(self, string ,tf_positive =0, idf_positive=0, tf_neutral=0, idf_neutral=0,tf_negative=0, idf_negative=0):
        self.string = string
        self.tf_positive = tf_positive
        self.tf_neutral = tf_neutral
        self.tf_negative = tf_negative
        self.idf_positive = idf_positive
        self.idf_neutral = idf_neutral
        self.idf_negative = idf_negative
        self.freq = 0
        self.sum_positive = 0
        self.sum_neutral = 0
        self.sum_negative = 0
        self.index = -1


if __name__ == "__main__":
    strings = ["a", "b", "c", "ab", "a"]
    words = set(strings)

    for word in words:
        print(word)