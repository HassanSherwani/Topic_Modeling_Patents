# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:24:29 2019

@Topic modeling for patent data
"""

# import basic modules
import pandas as pd
import numpy as np
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# Import GenSim and relevant modules
from gensim import corpora, models, similarities
from __future__ import print_function
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()

# A raw corpus of different patent abstracts related to electric vehicle and smart devices

raw_corpus = ["A self-propelled electric vehicle includes a wheeled frame having a quick connect and disconnect hitch for drivingly connecting the vehicle to a variety of wheeled devices. An individual drive for each of a pair of ground-contacting wheels includes a separate, reversible motor and a power transmission train coupled to each wheel which carries an inturned extension over which a transmission member is trained",
             "A robotic device has a base and at least one finger having at least two links that are connected in series on rotary joints with at least two degrees of freedom. A brushless motor and an associated controller are located at each joint to produce a rotational movement of a link. Wires for electrical power and communication serially connect the controllers in a distributed control network",
             "Method and system for remote monitoring of high-risk patients using artificial intelligence. A plurality of high-risk patients can be simultaneously monitored without patient intervention. A patient hears questions in the doctor's voice at each monitoring encounter and responds.The patient's responses are recorded at a remote central monitoring station and can be analyzed on line or later. ",
             "The utility model provides an electric automobile fills automatic robot in pond of discharging of getting of battery swapping station, a serial communication port, include: and a frame. Running gear sets up in the bottom of frame, is transverse movement, elevating system sets up in the centre of frame, is longitudinal motion, absorb battery mechanism, set up on elevating system, absorb the battery",              
             "An automated vehicle charging system, that may be done within a service type station, to provide for charging, recharging, or even discharging, of the batteries of an electric vehicle, and generally will include a dispenser, having a cabinet containing all of the instrumentation desired for furnishing the provision of current information relative to the charging of a vehicle",
             "This invention overcomes the disadvantages of the prior art by providing a human/machine interface (HMI) for use with machine vision systems (MVSs) that provides the machine vision system processing functionality at the sensor end of the system, and uses a communication interface to exchange control, image and analysis information with a standardized, preferably portable device that can be removed from the MVS during runtime",
             "A human-machine interface can detect when a user's ear is pulled back to initiate a plurality of procedures. Such procedures include turning on a TV using a laser attached to the user, starting an additional procedure by speaking a command, communicating with other users in environments which have high ambient noise, and interacting with the internet.",
             "The invention belongs to the technical field of automatic agricultural equipment, and particularly relates to a flexibly operable hand-eye mode spraying robot device which comprises a spray nozzle, a camera, a large mechanical arm, a small mechanical arm, a manipulator, a controller, a power source and a variable spray system.",
             "A relational artificial intelligence system is invented and developed. It comprises a relational automatic knowledge acquisition system and a relational reasoning system. The relational automatic knowledge acquisition system is a relational learning system which discovers knowledges from spreadsheet-formed databases and generates relational knowledge bases using inductive learning technique"]

type(raw_corpus)

# Cleaning and making Corpus

# Create a set of frequent words
stoplist = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in raw_corpus]

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
processed_corpus

from gensim import corpora
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)
print(dictionary.token2id)

# Creatng bag_of_words

bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
bow_corpus

from gensim import models
# train the model
tfidf = models.TfidfModel(bow_corpus)
# transform the "system minors" string
tfidf[dictionary.doc2bow("system minors".lower().split())]

# Transform

tfidf = models.TfidfModel(bow_corpus) # step 1 -- initialize a model
doc_bow = [(0, 1), (1, 1)]
print(tfidf[doc_bow]) # step 2 -- use the model to transform vectors

# First only vector is used. Now we need to use whole corpus
corpus_tfidf = tfidf[bow_corpus]
for doc in corpus_tfidf:
    print(doc)

#LSI Model

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
lsi.print_topics(2)
for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    print(doc)

#Validation part-Similarity matrix
    
doc = "electric vehicle robotics"
vec_bow = dictionary.doc2bow(doc.lower().split())
# convert the query to LSI space
vec_lsi = lsi[vec_bow]
print(vec_lsi)

# Initializing query structures
# transform corpus to LSI space and index it
index = similarities.MatrixSimilarity(lsi[corpus_lsi]) 

# Performing queries
# perform a similarity query against the corpus
sims = index[vec_lsi] 
# print (document_number, document_similarity) 2-tuples
print(list(enumerate(sims)))

sims = sorted(enumerate(sims), key=lambda item: -item[1])
# print sorted (document number, similarity score) 2-tuples
print(sims)


#Model 2:LDA 

tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                stop_words = 'english',
                                lowercase = True,
                                token_pattern = r'\b[a-zA-Z]{3,}\b',
                                )
dtm_tf = tf_vectorizer.fit_transform(raw_corpus)
print(dtm_tf.shape)

tfidf_vectorizer = TfidfVectorizer(**tf_vectorizer.get_params())
dtm_tfidf = tfidf_vectorizer.fit_transform(raw_corpus)
print(dtm_tfidf.shape)

# for TF DTM
lda_tf = LatentDirichletAllocation(n_topics=20, random_state=0)
lda_tf.fit(dtm_tf)
# for TFIDF DTM
lda_tfidf = LatentDirichletAllocation(n_topics=20, random_state=0)
lda_tfidf.fit(dtm_tfidf)

#Visualize the model

pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer)

pyLDAvis.sklearn.prepare(lda_tfidf, dtm_tfidf, tfidf_vectorizer)

pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer, mds='mmds')

pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer, mds='tsne')



