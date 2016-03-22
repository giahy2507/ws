import os
import numpy as np
import pickle
from sys import stdout
import time
import sys


class Vocabulary(object):
    def __init__(self, word_index, word_freq, alphabet, min_count):
        self.word_index = word_index
        self.embed_matrix = None
        self.word_freq = word_freq
        self.vocab_size = len(word_index.keys())
        self.alphabet = alphabet
        self.min_count = min_count

    def __str__(self):
        return "Vocabulary"

    def save(self, file):
        with open(file, mode="wb") as f:
            pickle.dump(self, f)

    def get_index(self, word):
        A = Vocabulary.str_intersection(word, self.alphabet)
        if len(A) == len(word):
            # push to dictionany, freq += 1
            if word in self.word_index.keys():
                index = self.word_index[word]
            else:
                index = 1
        else:
            index = 0

        return index

    @classmethod
    def load(cls, file):
        with open(file, mode="rb") as f:
            vocab = pickle.load(f)
            return Vocabulary(vocab.word_index, vocab.word_freq,  vocab.alphabet, vocab.min_count)

    @classmethod
    def filter_with_min_count(cls, word_freq, min_count):
        idx = 4
        word_index = {"symbol":0, "unk": 1, "head": 2, "tail":3}
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

    def rebuild_vocab_ws(self, train_filepath ,min_count = 5):
        # after rebuild wordindex will be changed
        print ("rebuild vocab")
        with open(train_filepath, mode="r") as fi:
            sentences = fi.readlines()
        for i, sentence in enumerate(sentences):
            if i % 100000 == 0:
                print("Rebuild vocab processed line : ", i)
            for word in sentence.replace("_"," ").lower().split():
                A = Vocabulary.str_intersection(word,self.alphabet)
                if len(A) == len(word):
                    # push to dictionany, freq += 1
                    if word not in self.word_freq.keys():
                        self.word_freq[word] = 1
                    else:
                        self.word_freq[word] +=1
                else:
                    self.word_freq["symbol"] +=1
        self.word_index = Vocabulary.filter_with_min_count(self.word_freq,min_count)

    @classmethod
    def build_vocab_ws(cls, train_filepath, alphatbet_filepath , min_count = 5):
        print ("Build vocab ... ")

        with open(train_filepath, mode="r") as fi:
            sentences = fi.readlines()

        word_freq  = {"symbol":0}

        alphabet = Vocabulary.load_alphabet(alphatbet_filepath)
        for i, sentence in enumerate(sentences):
            if i % 100000 == 0:
                print("Build vocab processed line ", i)
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

    def sen_2_index(self, sentence = "", tagset = "YN"):
        sentence_result = []
        tag_result = []
        words = sentence.split()
        for word in words:
            syllables = word.split('_')
            if len(syllables) == 0:
                print("exception: ", syllables)
                sys.exit(2)
            elif len(syllables) == 1:
                sentence_result.append(self.get_index(syllables[0]))
                if tagset == "YN":
                    tag_result.append(0)
                else:
                    print("Invalid tagset")
                    sys.exit(2)
            else:
                if tagset == "YN":
                    for syllable_idx in range(len(syllables)):
                        sentence_result.append(self.get_index(syllables[syllable_idx]))
                        if syllable_idx == len(syllables) -1:
                            tag_result.append(0)
                        else:
                            tag_result.append(1)
        return sentence_result, tag_result

def gen_data_from_sentence_indexs(words_index, labels):
    """
    Params:
        words_index:
        labels:
    :return: [[index_A,index_B,index_C,index_D,index E], ...] , [label_1, label2,]
    head: 2
    tail: 3
    [1 ,2 ,3]
    """
    if len(words_index) == 0:
        return np.array([[]]), np.array([])
    elif len(words_index) == 1:
        return np.array([[2,2,words_index[0],3,3]]),np.array(labels)
    elif len(words_index) == 2:
        return np.array([[2, 2, words_index[0], words_index[1], 3],
                         [2, words_index[0], words_index[1], 3, 3] ]),np.array(labels)
    elif len(words_index) == 3:
        return np.array([[2, 2, words_index[0], words_index[1], 3],
                         [2, words_index[0], words_index[1], words_index[2], 3],
                         [words_index[0], words_index[1],words_index[2], 3, 3]]),np.array(labels)
    elif len(words_index) == 4:
        return np.array([[2, 2, words_index[0], words_index[1], 3],
                         [2, words_index[0], words_index[1], words_index[2], words_index[3]],
                         [words_index[0], words_index[1],words_index[2], words_index[3], 3],
                         [words_index[1],words_index[2], words_index[3], 3, 3]]), np.array(labels)
    else:
        samples = []
        samples.append([ 2, 2, words_index[0], words_index[0+1], words_index[0+2]])
        samples.append([ 2, words_index[0], words_index[1], words_index[2], words_index[3]])
        for i in range(2,len(words_index)-2,1):
            samples.append(words_index[i-2:i+2+1])
        samples.append([ words_index[-4],words_index[-3], words_index[-2], words_index[-1], 3])
        samples.append([ words_index[-3], words_index[-2], words_index[-1], 3, 3])
        return np.array(samples), np.array(labels)

def build_vocab(file_path, save_path = "model/vocab.bin", alphabet_path = "rule/alphabet.txt", min_count=3):
    vocabulary = Vocabulary.build_vocab_ws(file_path,alphabet_path, min_count=3)
    # vocabulary.rebuild_vocab_ws("data/vcl.pre.txt", min_count=3)
    vocabulary.save(save_path)
    print(len(vocabulary.word_index.keys()))
    print(vocabulary.__str__())

if __name__ == "__main__":

    with open("model/vocab.bin", mode="rb") as f:
        vocabulary = pickle.load(f)
    print("Finish read Vocab")

    with open("data/vtb.pre.txt", mode="r") as f:
        lines = f.readlines()
    print("Finish read vtb")

    # [index_A index_B index_C index_D index E] ---------- label_C

    X = np.array([[0,0,0,0,0]])
    Y = np.array([0])

    start = time.clock()
    for i, line in enumerate(lines):
        if i % 10000 == 0:
            end = time.clock()
            sys.stdout.write("Prepare data from line "+ str( i - 10000) +  " to line " +str(i) +" : " + str( int(end - start) )+ "s\n")
            sys.stdout.flush()
            start = time.clock()
        A = line.replace("_"," ").split()
        sentence_indexs, tag_results = vocabulary.sen_2_index(line)

        if (len(sentence_indexs) != len(tag_results)):
            print("2 thang nay ko bang ne")
        else:
            xx, yy = gen_data_from_sentence_indexs(sentence_indexs,tag_results)
            X = np.concatenate((X,xx), axis=0)
            Y = np.concatenate((Y,yy))

    print(X.shape)
    print(Y.shape)
    print("Finish process vtb")

    fi = open("data/vcl.pre.txt", mode="r")
    print("Processing vtb")

    start = time.clock()
    for i, line in enumerate(fi):
        if i % 10000 == 0:
            end = time.clock()
            sys.stdout.write("Prepare data from line "+ str( i - 10000) +  " to line " +str(i) +" : " + str( int(end - start) )+ "s\n")
            sys.stdout.flush()
            start = time.clock()
        if i % 100000 == 0:
            valid_number = int(X.shape[0]*0.8)
            X_train, Y_train, X_validation, Y_validation = X[:valid_number], Y[:valid_number], X[valid_number:], Y[valid_number:]
            with open("data/vtb.pre.txt.train.nparray", mode="wb") as f:
                pickle.dump((X_train, Y_train, X_validation, Y_validation),f)
            print("Saved data to ", "data/vtb.pre.txt.train.nparray")
            start = time.clock()
        A = line.replace("_"," ").split()
        sentence_indexs, tag_results = vocabulary.sen_2_index(line)

        if (len(sentence_indexs) != len(tag_results)):
            print("2 thang nay ko bang ne")
        else:
            xx, yy = gen_data_from_sentence_indexs(sentence_indexs,tag_results)
            X = np.concatenate((X,xx), axis=0)
            Y = np.concatenate((Y,yy))
    fi.close()
    print("Finish process vcl")

    valid_number = int(X.shape[0]*0.8)
    X_train, Y_train, X_validation, Y_validation = X[:valid_number], Y[:valid_number], X[valid_number:], Y[valid_number:]
    with open("data/vtb.pre.txt.train.nparray", mode="wb") as f:
        pickle.dump((X_train, Y_train, X_validation, Y_validation),f)
    print("Saved data to ", "data/vtb.pre.txt.train.nparray")













