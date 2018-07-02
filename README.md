# Legal-Sents-Embs
This project is to construct sentence embeddings for legal language and analyze semantic similarity between two sentences.
The code is based on [the paper](https://openreview.net/forum?id=SyK00v5xx) "A Simple but Tough-to-Beat Baseline for Sentence Embeddings".

The code is written in Python and it needs numpy, pandas, matplotlib, scipy, pickle, sklearn, gensim.

## Install

To install all dependencies, `virtualenv` is suggested:

```
$ virtualenv .env
$ . .env/bin/activate
$ pip install -r requirements.txt 
```

## Comments for Demos

In the folder *examples*,

* Generate_Basic_Files_WE.py is to generate words file, word embeddings file, and weight4ind file from the pretrained word embeddings file, such as fastText or GloVe.

* Generate_Basic_Files_Gensim.py is to generate words file, word embeddings file, and weight4ind file from the word embeddings in the gensim output format.

* sif_embedding.py is to generate a sentence embedding given a sentence.

* sim_sif.py is to generate sentence embeddings for the given .txt file, and outputs the Pearson's Coefficent and MSE.

* generate_first_pc.py is to compute the first principal component given a set of sentences.

## Get Started

The sentence embeddings is based on the pre-trained word embeddings library, such as word2vec, GloVe, or fastText. 

Firstly, using *Generate_Basic_Files_WE.py* or *Generate_Basic_Files_Gensim.py* to generate three necessary files for the further step, namely, words, vectors, and weight4ind. The source file of the word embeddings libraries and the output location should be passed into the main function.

Secondly, using *generate_first_pc.py* to generate the first principal component of a set of sentences in the dataset.

Thirdly, using *sif_embedding.py* to generate a sentence embedding for the givne sentence. Or, using *sim_sif.py* to produce the evaluation result (Pearson's Coefficient and MSE) on the test dataset.

More details can be found in the code with the comments.
