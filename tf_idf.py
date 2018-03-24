import os
import pickle
from random import shuffle, seed
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

"""
Reads txt files of all papers and computes tfidf vectors for all papers.
Dumps results to file tfidf.p

Code inspired by: https://github.com/karpathy/arxiv-sanity-preserver/blob/master/analyze.py
"""
seed(1337)
#max_features = None
max_features = 100000

id_to_filename = json.load(open('./extraction/id_to_filename.json', 'r'))
data_path = './extraction/data/'
tfidf_path = './tfidf.json'
meta_path = './meta_tfidf.json'
sim_path = './sim_dict.json'

N = len(id_to_filename)

# read all text files for all papers into memory 
# (to filter out short and long videos)
good_txt_paths = []
good_id_to_filename = {}
n = 0
for id in id_to_filename:
  n += 1
  filename = id_to_filename[id]
  txt_path = data_path+filename+'.txt'
  with open(txt_path, 'r') as f:
    txt = f.read()
    if len(txt) > 1000 and len(txt) < 500000: # 500K is VERY conservative upper bound
      good_txt_paths.append(txt_path) 
      good_id_to_filename[id] = id_to_filename[id]
      print("read {}/{} ({}) with {} chars".format(n, len(id_to_filename), filename, len(txt)))
    else: 
      print("skipped {}/{} ({}) with {} chars".format(n, len(id_to_filename), filename, len(txt))) 
print()
print("in total read in {} text files out of {} db entries.".format(len(good_txt_paths), len(id_to_filename)))

# compute tfidf vectors with scikit
vectorizer = TfidfVectorizer(input='content', 
        encoding='utf-8', decode_error='replace', strip_accents='unicode', 
        lowercase=True, analyzer='word', stop_words='english', 
        token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
        ngram_range=(1, 2), max_features=max_features, 
        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True, min_df=2)

# create an iterator object to conserve memory
def make_corpus(paths):
  """iterator to yield text of all files individually based on list of paths"""
  for p in paths:
    with open(p, 'r') as f:
      txt = f.read()
    yield txt

# transform
print("transforming {} documents...".format(len(good_txt_paths), ))
corpus = make_corpus(good_txt_paths)
X = vectorizer.fit_transform(corpus)
print(vectorizer.vocabulary_)
print(X.shape)
# distance matrix
D = (X * X.T).todense()
# raw tf-idfs
X = X.todense()

# write full matrix out
out = {}
# these are heavy!
#out['X'] = X.tolist() 
out['D'] = D.tolist()
print("writing", tfidf_path)
json.dump(out, open(tfidf_path, 'w'))

# writing lighter metadata information into a separate (smaller) file
out = {}
#vocab = {key: vectorizer.vocabulary_[key].tolist() for key in vectorizer.vocabulary_.keys()}
#out['vocab'] = vocab
#out['idf'] = vectorizer._tfidf.idf_.tolist()
out['ids'] = good_id_to_filename
out['itoid'] = {i:x for i,x in enumerate(good_id_to_filename.keys())}
print("writing", meta_path)
json.dump(out, open(meta_path, 'w'))

#print("precomputing nearest neighbor queries in batches...")
#sim_dict = {}
#batch_size = 200
#ids = list(good_id_to_filename.keys())
#for i in range(0,len(good_txt_paths),batch_size):
#  i1 = min(len(good_txt_paths), i+batch_size)
#  xquery = X[i:i1] # BxD
#  ds = -np.asarray(np.dot(X, xquery.T)) #NxD * DxB => NxB
#  IX = np.argsort(ds, axis=0) # NxB
#  for j in range(i1-i):
#    sim_dict[ids[i+j]] = [ids[q] for q in list(IX[:50,j])]
#  print('{}/{}...'.format(i, len(out['ids'])))
#
#print("writing", sim_path)
#json.dump(sim_dict, open(sim_path, 'w'))

print("Done.")