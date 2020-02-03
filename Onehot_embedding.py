import numpy as np
import re

f=open("train_tag_dict.txt","r")
data =eval(f.read())


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 0

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            if word != "hair" and word != "eyes" and word != "and":
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # 删除低于特定计数阈值的词，降低模型复杂度
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(len(keep_words), len(self.word2index),
                                                   len(keep_words) / len(self.word2index)))

        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 0

        for word in keep_words:
            self.addWord(word)

    def split_description(self, sentence):
        q = sentence.split(" ")
        hair = []
        eyes = []
        if "and" in q:
            del q[q.index("and")]
            hair_index = q.index("hair")
            eyes_index = q.index("eyes")
            if hair_index < eyes_index:
                hair = q[0:hair_index]
                eyes = q[hair_index + 1:eyes_index]
            else:
                eyes = q[0:eyes_index]
                hair = q[eyes_index + 1:hair_index]

            return hair, eyes

        if "hair" in q:
            del q[q.index("hair")]
            hair = q
        if "eyes" in q:
            del q[q.index("eyes")]
            eyes = q

        return hair, eyes

    def generate_2_vectors(self, hair, eyes):
        h_vector = np.zeros(self.num_words)
        e_vector = np.zeros(self.num_words)

        for i in hair:
            h_vector[self.word2index[i]] = 1
        for i in eyes:
            e_vector[self.word2index[i]] = 1

        return h_vector, e_vector

    def generate_final_vector(self,description):
        hair,eyes = self.split_description(description)
        h_vector,e_vector = self.generate_2_vectors(hair,eyes)
        return np.concatenate((h_vector,e_vector),axis=0)



voc = Voc("d_vector")
for key in data:
    voc.addSentence(data[key])
voc.trim(100)

import os
#删除未知标签的文件

def delete_imgs(data):
    required_img = list(data.keys())

    for i in range(33430):
        i=i+1
        if i not in required_img:
            path = r"./faces/"+str(i)+".jpg"
            os.remove(path)