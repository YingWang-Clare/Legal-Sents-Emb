import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys
sys.path.append('../src')
import data_io
import sim_algo
import eval
import params
import os
from os import listdir
from os.path import isfile, join


def main(words_path, vectors_path, weight_path, fpc_name):
	rmpc = 1
	params = params.params()
	params.rmpc = rmpc

	# Loading preprocessed words, vectors and weight4ind files.
    print("loading words file...")
    words = pickle.load(open(words_path, 'rb'))
    print("loading vectors file...")
    vectors = pickle.load(open(vectors_path, 'rb'))
    print("loading weight4ind file...")
    weight4ind = pickle.load(open(weight_path, 'rb'))

    # Using a list of datasets to generate the corresponding fpc files.
	dataset_dir = "../data/"
	dataset_list = [f for f in listdir(dataset_dir) if isfile(join(dataset_dir, f))]
	for dataset_file in dataset_list:
	    print("preparing the first principle component based on {}.".format(str(dataset_file)))
	    eval.prepare_first_pc(vectors, words, weight4ind, sim_algo.get_first_pc, params, dataset_file)

	test_dataset = 'sicktest' # name of the test dataset
	pearson_list = []
	mse_list = []
	index = [fpc for fpc in fpc_list]

	# Using a list of fpc files to evaluate on datasets.
	fpc_dir = "../first_principle_component/"
	fpc_list = [f for f in listdir(fpc_dir) if isfile(join(fpc_dir, f))]
	for fpc_file in fpc_list:
	    print("calculating sentence similarity scores, use fpc file: {}.".format(fpc_file))
	    pearson, mse = eval.sim_evaluate_one(vectors, words, weight4ind, sim_algo.weighted_average_sim_rmpc, params, fpc_file, test_dataset)
	    pearson_list.append(pearson)
	    mse_list.append(mse)

if __name__ == '__main__':
    plac.call(main)
