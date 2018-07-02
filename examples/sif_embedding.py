import sys
import pickle
import os
import plac
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../src')


@plac.annotations(
    words_path=("Location of words file"),
    vectors_path=("Location of vectors file"),
    weight_path=("Location of weight4ind file"),
    fpc_name=("File name of first_principal_component file")
)
def SIF_embedding(We, x, w, fpc_file):
    emb = get_weighted_average(We, x, w)
    emb = remove_pc(emb, fpc_file)
    return emb


def get_weighted_average(We, x, w):
    We = np.array(We)
    print("the shape of We is {}".format(np.shape(We)))
    n_samples = x.shape[0]
    print("the number of samples is {}".format(n_samples))
    emb = np.zeros((n_samples, We.shape[1]))
    for i in xrange(n_samples):
        emb[i, :] = w[i, :].dot(We[x[i, :], :]) / np.count_nonzero(w[i, :])
    return emb


def remove_pc(X, fpc_file):
    pc = load_pc(fpc_file)
    print("pc loaded has shape {}".format(np.shape(pc)))
    XX = X - X.dot(pc.transpose()) * pc
    return XX


def load_pc(filename):
    path2file = os.path.join("../first_principal_component/", filename)
    return np.load(path2file)


def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask


def seq2weight(seq, mask, weight4ind):
    weight = np.zeros(seq.shape).astype('float32')
    for i in xrange(seq.shape[0]):
        for j in xrange(seq.shape[1]):
            if mask[i, j] > 0 and seq[i, j] >= 0:
                weight[i, j] = weight4ind[seq[i, j]]
    weight = np.asarray(weight, dtype='float32')
    return weight


def getSeq(p1, words):
    p1 = p1.split()
    X1 = []
    for i in p1:
        X1.append(lookupIDX(words, i))
    return X1


def lookupIDX(words, w):
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#", "")
    if w in words:
        return words[w]
    elif 'UUUNKKK' in words:
        return words['UUUNKKK']
    else:
        return len(words) - 1


def main(words_path, vectors_path, weight_path, fpc_name):
    # Loading preprocessed words, vectors and weight4ind files.
    print("loading words file...")
    words = pickle.load(open(words_path, 'rb'))
    print("loading vectors file...")
    vectors = pickle.load(open(vectors_path, 'rb'))
    print("loading weight4ind file...")
    weight4ind = pickle.load(open(weight_path, 'rb'))

    # Generating a sentence embedding for one sentence.
    sentences = 'The Board of Directors may cause the Partnership to purchase or otherwise acquire Partnership Interests  provided  however  that the Board of Directors may not cause any Group Member to purchase Subordinated Units during the Subordination Period'
    fpc_file = fpc_name

    X = getSeq(sentences, words)
    seq = []
    seq.append(X)
    x, m = prepare_data(seq)
    w = seq2weight(x, m, weight4ind)
    embedding = SIF_embedding(We, x, w, fpc_file)
    embedding = np.squeeze(embedding)
    embedding = embedding.tolist()
    return embedding


if __name__ == '__main__':
    plac.call(main)
