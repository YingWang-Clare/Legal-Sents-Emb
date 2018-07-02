from django.utils.encoding import smart_str
from gensim.models import Word2Vec, KeyedVectors
from decimal import Decimal
import plac
import pickle
import os


@plac.annotations(
    gensim_model_path=("Location of gensim's .bin file"),
    out_dir=("Location of output directory"),
)
def main(gensim_model_path, out_dir):
    gensim_model = Word2Vec.load(gensim_model_path)
    words = {}
    n = 0
    vectors = []
    weightfile_name = os.path.join(out_dir, "weightfile.txt")
    weightfile = open(weightfile_name, "w")
    for string in gensim_model.wv.vocab:
        vocab = gensim_model.wv.vocab[string]
        freq, idx = vocab.count, vocab.index
        weightfile.write(smart_str(string))
        weightfile.write(" ")
        weightfile.write(smart_str(freq))
        weightfile.write("\n")
        vector = gensim_model.wv.syn0[idx]
        vectors.append(vector)
        words[string] = n
        n = n + 1

    vector_file = open(os.path.join(out_dir, "vectors"), "w")
    pickle.dump(vectors, vector_file)

    words_file = open(os.path.join(out_dir, "words"), "w")
    pickle.dump(words, words_file)

    weightpara = [1e-2, 1e-3, 1e-4]
    for a in weightpara:
        print("calculating word2weight with a = {}.".format(a))
        word2weight = getWordWeight(weightfile_name, a)
        print("calculating weight4ind with a = {}.".format(a))
        weight4ind = getWeight(words, word2weight)
        weight4ind_file = open(os.path.join(out_dir, "weight4ind_weightpara_%.E" % Decimal(a)), 'w')
        pickle.dump(weight4ind, weight4ind_file)


def getWordWeight(weightfile, a=1e-3):
    if a <= 0:  # when the parameter makes no sense, use unweighted
        a = 1.0

    word2weight = {}
    with open(weightfile) as f:
        lines = f.readlines()
    N = 0
    for i in lines:
        i = i.strip()
        if (len(i) > 0):
            i = i.split()
            if (len(i) == 2):
                word2weight[i[0]] = float(i[1])
                N += float(i[1])
            else:
                print(i)
    for key, value in word2weight.iteritems():
        word2weight[key] = a / (a + value / N)
    return word2weight


def getWeight(words, word2weight):
    weight4ind = {}
    for word, ind in words.iteritems():
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0
    return weight4ind


if __name__ == '__main__':
    plac.call(main)
