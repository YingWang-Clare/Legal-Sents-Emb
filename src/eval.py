import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from math import sqrt
from pylab import figure
import numpy as np
import os
import data_io
import sim_algo
import pickle

################################################
# for textual similarity tasks
################################################


def sim_getCorrelation(We, words, f, weight4ind, scoring_function, params, fpc, test_name):
    f = open(f, 'r')
    lines = f.readlines()
    golds = []
    seq1 = []
    seq2 = []
    index = []
    idx = 0
    for i in lines:
        i = i.split("\t")
        p1 = i[0]
        p2 = i[1]
        score = float(i[2])
        X1, X2 = data_io.getSeqs(p1, p2, words)
        seq1.append(X1)
        seq2.append(X2)
        golds.append(score)
        index.append(idx)
        idx += 1
    x1, m1 = data_io.prepare_data(seq1)
    x2, m2 = data_io.prepare_data(seq2)
    m1 = data_io.seq2weight(x1, m1, weight4ind)
    m2 = data_io.seq2weight(x2, m2, weight4ind)
    golds = np.asarray(golds)
    scores = scoring_function(We, x1, x2, m1, m2, params, fpc)
    # scores = scoring_function(We, x1, x2, m1, m2, golds, params, fpc)
    # preds = np.squeeze(scores).reshape(-1, 1)
    preds = np.squeeze(scores)
    # print('the prediction list is {}'.format(preds))

    # add SVM predictor
    # clf = pickle.load(open('../score_predictor/model_svm', 'rb'))
    # clf.fit(preds, golds)
    # preds = clf.predict(preds)

    print(preds)
    # np.save(open("../pred_list", 'wb'), preds)
    # np.save(open("../gold_list", 'wb'), golds)
    # show_result_image(preds, golds, index, fpc, test_name)
    # find_bad_scores(preds.tolist(), lower_threshold=2.5, higher_threshold=3.8)
    MSE = sqrt(mean_squared_error(golds, preds))
    return pearsonr(preds, golds)[0], MSE

def sim_badSents(We, words, weight4ind, scoring_function, params, fpc, sent1, sent2):
    seq1 = []
    seq2 = []

    X1, X2 = data_io.getSeqs(sent1, sent2, words)
    seq1.append(X1)
    seq2.append(X2)

    x1, m1 = data_io.prepare_data(seq1)
    x2, m2 = data_io.prepare_data(seq2)
    m1 = data_io.seq2weight(x1, m1, weight4ind)
    m2 = data_io.seq2weight(x2, m2, weight4ind)
    scores = scoring_function(We, x1, x2, m1, m2, params, fpc)
    preds = np.squeeze(scores)
    preds = preds * 2 + 3
    return preds

def find_bad_scores(list_of_score, lower_threshold, higher_threshold):
    lower = 0
    higher = 0
    for score in list_of_score:
        if score < lower_threshold:
            lower += 1
        if score >= higher_threshold:
            higher += 1
    print("the number of scores lower than {} is {} out of {}.".format(lower_threshold, lower, len(list_of_score)))
    print("the number of scores higher than {} is {} out of {}.".format(higher_threshold, higher, len(list_of_score)))

def show_result_image(prediction, truth_label, index, fpc_name, test_name):
    plt.figure(2)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 30
    fig_size[1] = 6
    plt.plot(index, prediction, label='predictions')
    plt.plot(index, truth_label, label='ground truth label')
    plt.xlabel('pairs index')
    plt.ylabel('scores')
    plt.legend()
    image_name = fpc_name + "_" + test_name + ".png"
    save_path = os.path.join("../results_image/", image_name)
    plt.savefig(save_path)

def prepare_first_pc(We, words, weight4ind, generation_function, params, fpc):
    print("reading file: {}.".format(fpc))
    # pre_calculate_first_pc(We, words, fpc, weight4ind, generation_function, params)
    file_name = fpc
    f = os.path.join("../data/", fpc)
    f = open(f, 'r')
    seq = []
    for i in f.readlines():
        X = data_io.getSeq(i, words)
        seq.append(X)
    x, m = data_io.prepare_data(seq)
    m = data_io.seq2weight(x, m, weight4ind)
    generation_function(We, x, m, params, file_name)


def sim_evaluate_one(We, words, weight4ind, scoring_function, params, fpc, test):
    path2test = os.path.join("../data/", test)
    pearson, mse = sim_getCorrelation(We, words, path2test, weight4ind, scoring_function, params, fpc, test)
    print("Test on dataset: {}.".format(test))
    print("Pearson Correlation = {}, MSE = {}.".format(pearson, mse))
    return pearson, mse