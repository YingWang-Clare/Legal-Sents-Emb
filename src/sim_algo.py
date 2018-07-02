import numpy as np
import SIF_embedding
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import pickle, os

def weighted_average_sim_rmpc(We, x1, x2, w1, w2, params, fpc_file):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x1: x1[i, :] are the indices of the words in the first sentence in pair i
    :param x2: x2[i, :] are the indices of the words in the second sentence in pair i
    :param w1: w1[i, :] are the weights for the words in the first sentence in pair i
    :param w2: w2[i, :] are the weights for the words in the first sentence in pair i
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: scores, scores[i] is the matching score of the pair i
    """
    emb1 = SIF_embedding.SIF_embedding(We, x1, w1, params, fpc_file)
    emb2 = SIF_embedding.SIF_embedding(We, x2, w2, params, fpc_file)
    print(emb1)
    print(emb2)
    # only cosine distance
    inn = (emb1 * emb2).sum(axis=1)
    emb1norm = np.sqrt((emb1 * emb1).sum(axis=1))
    emb2norm = np.sqrt((emb2 * emb2).sum(axis=1))
    scores = inn / emb1norm / emb2norm
    return scores

def weighted_average_sim_rmpc_linear_regression(We,x1,x2,w1,w2, golds, params, fpc_file):

    emb1 = SIF_embedding.SIF_embedding(We, x1, w1, params, fpc_file)
    emb2 = SIF_embedding.SIF_embedding(We, x2, w2, params, fpc_file)
    # cosine distance & euclidean distance
    inn = (emb1 * emb2).sum(axis=1)
    emb1norm = np.sqrt((emb1 * emb1).sum(axis=1))
    emb2norm = np.sqrt((emb2 * emb2).sum(axis=1))
    cos_dist = inn / emb1norm / emb2norm
    cos_dist = (cos_dist / np.amax(cos_dist)).reshape(-1, 1)
    euc_dist = np.linalg.norm(emb1 - emb2, axis=1)
    euc_dist = (euc_dist / np.amax(euc_dist)).reshape(-1, 1)
    # scores = 0.8 * cos_dist + 0.2 * euc_dist
    vec_dist = np.concatenate((cos_dist, euc_dist), axis=1)

    # linear regression predictor
    # predictor = LinearRegression()
    # predictor.fit(vec_dist, golds)
    # scores = predictor.predict(vec_dist)
    # pickle.dump(predictor, open('../score_predictor/model_linear_regression', 'wb'))
    # predictor = pickle.load(open('../score_predictor/model_linear_regression', 'rb'))
    # scores = predictor.predict(vec_dist)

    # SVM predictor
    # predictor = SVR(C=2, epsilon=0.05)
    # predictor.fit(vec_dist, golds)
    # scores = predictor.predict(vec_dist)
    # pickle.dump(predictor, open('../score_predictor/svm_vec_dis', 'wb'))
    # print(mean_absolute_error(golds, scores))
    return scores

def get_first_pc(We, x, w, params, fpc_file):
    SIF_embedding.calculate_first_pc(We, x, w, params, fpc_file)

def get_second_pc(We, x, w, params, fpc_file):
    SIF_embedding.calculate_second_pc(We, x, w, params, fpc_file)