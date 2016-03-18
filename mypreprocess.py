__author__ = 'HyNguyen'

import re
from openpyxl import load_workbook

class Preprocess(object):
    def __init__(self, alphabet,split_word, replace_dict, emoticon = None):
        self.alphabet = alphabet
        self.replace_dict = replace_dict
        self.split_word = split_word
        self.emoticon = emoticon

    def __str__(self):
        return "Preprocessing Assistant"

    @classmethod
    def create_from_rule_file(cls, alphabet_path, replace_dict_path, split_word_path):
        with open(alphabet_path, mode="r") as f:
            alphabet = f.readline()
        with open(split_word_path, mode="r") as f:
            split_word = f.readline()
        with open(replace_dict_path, mode="r") as f:
            lines = f.readlines()
            replace_dict = {}
            for line in lines:
                line = line.replace("\n","")
                replaced_word, word = line.split("\t")
                replace_dict[replaced_word.lower()] = "#"+word.lower()
        return Preprocess(alphabet,split_word,replace_dict)

    def replace_word(self, sentence):
        for key in self.replace_dict.keys():
            sentence = sentence.replace(key, self.replace_dict[key])
        return sentence

    def replace_phone(self, sentence):
        sentence = re.sub(r"(\d{2,5})[ .](\d{2,5})[ .](\d{2,5})", " #phone ", sentence)
        sentence = re.sub(r"(\+\d{2,4})?[ .]?(\d{8,11})", " #phone ", sentence)
        sentence = re.sub(r"(\+\d{2,4})[ .](\d{2,3})[ .](\d{7})", " #phone ", sentence)
        return sentence

    def replace_hastag(self, sentence):
        sentence = re.sub("#([\S]+)", " #hastag ", sentence)
        return sentence

    def replace_date(self, sentence):
        sentence = re.sub(r"([0-9]+)[/]([0-9]+)[/]([0-9]+)", " #date ", sentence)
        sentence = re.sub(r"([0-9]+)[/]([0-9]+)", " #date ", sentence)
        return sentence

    def replace_number(self, sentence):
        sentence = re.sub(r"[-+]?([0-9]*\.[0-9]+|[0-9]+)", " #number ", sentence)
        return sentence

    def islink(self, word):
        #only heuristic
        result = False
        #1. contain these below words
        m = 0
        heuristic = ['.com','http://','www','http','.vn','.edu','.en','https']
        for h in heuristic:
            if (h in word):
                result = True
                m += 1
        #2. Number of character '/' + number of heuristic word appear in word is bigger than max_number = 3
        n = word.count('/') + m
        if (n >= 3 and result == True):
            return True
        else:
            return False

    def replace_link(self, sentence):
        words = sentence.split()
        for i in range(len(words)):
            if (self.islink(words[i]) == True):
                words[i] = "#URL"
        return " ".join(words)

    # split ra truoc sau do moi check alphabet
    # check alphabet de cuoi cung
    def split_word_by_alphabet_splitword(self, word, mode):
        # if mode = True, split symbol in head and tail ele of word
        # if mode = False, split symbol in all word
        if (mode == True):
            if ((word[0] in self.split_word ) == True):
                word = word[0] + ' ' + word[1:]
            if ((word[-1] in self.split_word) == True):
                word = word[0:len(word)-1] + ' ' + word[-1]
        else:
            i = 0
            while (i < len(word)):
                stop = False
                if ((word[i] in self.split_word) == True):
                    word = word[0:i] + ' ' + word[i] + ' ' + word[i+1:len(word)]
                    i += 1
                i += 1
            if ((word[-1] in self.split_word) == True and word[-2] in self.alphabet):
                word = word[0:len(word)-1] + ' ' + word[-1]
                return word
        return word

    def preprocess_sentence(self, sentence):
        sentence = sentence.lower()
        sentence = self.replace_word(sentence)
        sentence = self.replace_hastag(sentence)
        sentence = self.replace_date(sentence)
        sentence = self.replace_phone(sentence)
        sentence = self.replace_number(sentence)
        sentence = self.replace_link(sentence)
        words = sentence.split()
        for i in range(len(words)):
            words[i] = self.split_word_by_alphabet_splitword(words[i], False)
        sentence = " ".join((" ".join(words)).split())
        return sentence

import argparse

if __name__ == "__main__":
    '''
    usage: mypreprocess.py [-h] -fi FI -fo FO
    '''
    parser = argparse.ArgumentParser(description='Parse process ')
    parser.add_argument('-fi', required=True, type = str)
    parser.add_argument('-fo', required=True, type = str)
    args = parser.parse_args()

    dir_input = args.fi
    dir_ouput = args.fo

    preprocess = Preprocess.create_from_rule_file("rule/alphabet.txt", "rule/exceptword.txt","rule/splitword.txt")
    with open(dir_input, mode="r") as f:
        lines = f.readlines()
    file_out = open(dir_ouput,mode="w")
    for i,line in enumerate(lines):
        if i % 100000 == 0:
            print("Preprocessed ", i, "line")
        sentence = preprocess.preprocess_sentence(line)
        file_out.write(sentence + "\n")
    file_out.close()
    print(preprocess.__str__())