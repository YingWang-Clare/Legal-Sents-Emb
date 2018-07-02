import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys
import os
import plac
sys.path.append('../src')
from os import listdir
from os.path import isfile, join
import data_io
import sim_algo
import eval
import params


@plac.annotations(
    words_path=("Location of words file"),
    vectors_path=("Location of vectors file"),
    weight_path=("Location of weight4ind file"),
    fpc_name=("File name of first_principal_component file"),
    test_name=("File name of the test dataset")
)
def main(words_path, vectors_path, weight_path, fpc_name, test_name):
    # Loading preprocessed words, vectors and weight4ind files.
    print("loading words file...")
    words = pickle.load(open(words_path, 'rb'))
    print("loading vectors file...")
    vectors = pickle.load(open(vectors_path, 'rb'))
    print("loading weight4ind file...")
    weight4ind = pickle.load(open(weight_path, 'rb'))
        rmpc = 1
        params = params.params()
        params.rmpc = rmpc
        fpc_file = fpc_name
        test_dataset = test_name
        print("calculating sentence similarity scores, use fpc file: {}.".format(fpc_file))
        pearson, mse = eval.sim_evaluate_one(vectors, words, weight4ind, sim_algo.weighted_average_sim_rmpc, params, fpc_file, test_dataset)
