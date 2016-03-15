import os
import numpy as np
import pickle

class Vocabulary(object):
    def __init__(self, word_index, word_freq, alphabet, min_count):
        self.word_index = word_index
        self.embed_matrix = None
        self.word_freq = word_freq
        self.load_alphabet = alphabet
        self.min_count = min_count

    @classmethod
    def filter_with_min_count(cls, word_freq, min_count):
        idx = 1
        word_index = {"symbol":0}
        for word in word_freq.keys():
            if word_freq[word] >= min_count:
                word_index[word] = idx
                idx+=1
        return word_index

    @classmethod
    def load_alphabet(cls, filename):
        fi = open(filename, "r")
        alphabet = fi.readline()
        fi.close()
        return alphabet

    @classmethod
    def str_intersection(cls, s1, s2):
        out = ""
        for c in s1:
            if c in s2:
                out += c
        return out

    @classmethod
    def build_vocab_ws(cls, sentences, alphatbet_filepath , min_count = 5):
        print ("build vocab")
        word_freq  = {"symbol":0}
        alphabet = Vocabulary.load_alphabet(alphatbet_filepath)
        idx = 1
        for i, sentence in enumerate(sentences):
            if i % 1000 == 0:
                print("processed line :", i)
            for word in sentence.replace("_"," ").lower().split():
                A = Vocabulary.str_intersection(word,alphabet)
                if len(A) == len(word):
                    # push to dictionany, freq += 1
                    if word not in word_freq.keys():
                        word_freq[word] = 1
                    else:
                        word_freq[word] +=1
                else:
                    word_freq["symbol"] +=1
        word_index = Vocabulary.filter_with_min_count(word_freq,min_count)
        return Vocabulary(word_index,word_freq,alphabet,min_count)

if __name__ == "__main__":
    alphabet =  Vocabulary.load_alphabet("rule/alphabet.txt")
    fi = open("train-file.txt", mode="r")
    lines = fi.readlines()
    fi.close()
    fi = open("train-file.txt", mode="r")
    lines += fi.readlines()
    fi.close()
    max_id = 0
    vocab = Vocabulary.build_vocab_ws(lines,"rule/alphabet.txt",min_count=3)
    for word in vocab.word_index.keys():
        if vocab.word_index[word] > max_id:
            max_id = vocab.word_index[word]
        print(word," ", vocab.word_index[word])




