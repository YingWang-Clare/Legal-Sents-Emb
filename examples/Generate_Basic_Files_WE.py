import pickle
import sys
import os
from decimal import Decimal
import plac
sys.path.append('../src')
import data_io
import sim_algo
import eval
import params
import numpy as np


@plac.annotations(
    word_embeddings_path=("Location of pre-trained word embeddings .txt file"),
    word_weight_path=("Location of the word weights .txt file"),
    out_dir=("Location of output directory"),
)
def getWordmap(textfile):
    words = {}
    We = []
    f = open(textfile, 'r')
    lines = f.readlines()
    for (n, i) in enumerate(lines):
        i = i.split()
        j = 1
        standard = 301
        if len(i) != standard:
            continue
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]] = n
        v = np.array(v)
        We.append(v)
        if v.shape[0] != (standard - 1):
            print("shape of v {}".format(v.shape))
            print("shape of We {}".format(np.array(We).shape))
    We = np.array(We)
    print('type of we', type(We))
    return (words, We)


def main(word_embeddings_path, word_weight_path, out_dir):
    wordfile = word_embeddings_path
    weightfile = word_weight_path
    weightparas = [1e-2, 1e-3, 1e-4]
    (words, We) = getWordmap(wordfile)
    vector_file = open(os.path.join(out_dir, "vectors"), "w")
    pickle.dump(We, vector_file)
    words_file = open(os.path.join(out_dir, "words"), "w")
    pickle.dump(words, words_file)
    for weightpara in weightparas:
        print("calculating word2weight with a = {}.".format(weightpara))
        word2weight = data_io.getWordWeight(weightfile, weightpara)
        print("calculating weight4ind with a = {}.".format(weightpara))
        weight4ind = data_io.getWeight(words, word2weight)
        weight4ind_file = open(os.path.join(out_dir, "weight4ind_weightpara_%.E" % Decimal(weightpara)), 'w')
        pickle.dump(weight4ind, weight4ind_file)


if __name__ == '__main__':
    plac.call(main)
