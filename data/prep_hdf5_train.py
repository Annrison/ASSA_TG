#!/usr/bin/env python

import numpy as np
import h5py
import re
import sys
import operator
import argparse
from random import sample, seed
from math import ceil
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
from gensim.models import Word2Vec
import pandas as pd
import os

# 每隔n切一個chunk，這裡的n就是之後的batch size
def parallel_chunks(l1, l2, l3, n):
    """
    Yields chunks of size n from 3 lists in parallel
    """
    if len(l1) != len(l2) or len(l2) != len(l3):
        raise IndexError
    else:
        for i in range(0, len(l1), n):
            yield l1[i:i+n], l2[i:i+n], l3[i:i+n]

def load_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    model = Word2Vec.load(fname)
    for v in vocab:
        try:
            word_vecs[v] = model.wv[v]
        except:
            word_vecs[v] = np.zeros(300, dtype='float32')
    return word_vecs

def line_to_words(line, min_len, max_len, stop_words=None):
    """
    Reads a line of text (sentence) and returns a list of tokenized EDUs
    將句子斷詞，篩選長度太長的句子（句子長度介於 min_len 與 max_len 之間）
    """

    # word_list = nltk.word_tokenize(line)    
    word_list = line.split(' ')
    if stop_words is not None:
        words = [word for word in word_list if word not in stop_words]

    # discard short segments
    if len(words) < min_len:
        # print("句子太短",len(words),words)
        return None
    
    # truncate long ones
    if len(words) > max_len:
        print("句子太長",len(words))
        words = words[:max_len]

    return words


def get_vocab(file, min_len, max_len, stop_words):
    """
    Reads an input file and builds vocabulary, product mapping, etc.
    """
    max_len_actual = 0
    wid = 1
    word2id = {}
    word2cnt = {}
    doc_cnt = 0

    train_df = pd.read_csv(file)
    for index, row in train_df.iterrows():        
        sentence = row['sentence']
        words = line_to_words(sentence, min_len, max_len, stop_words)  

        if words is None:
            # print("skip",row['scode'])
            continue      
        for word in words:
            max_len_actual = max(max_len_actual, len(word))
            if word not in word2id:
                word2id[word] = wid
                wid += 1
            if word not in word2cnt:
                word2cnt[word] = 1
            else:
                word2cnt[word] += 1
        doc_cnt += 1
    return max_len_actual, doc_cnt, word2id, word2cnt

# 注意如果句子裡都是停用字，就會被刪掉
def load_data(file, stop_words):
    """
    Loads dataset into appropriate data structures
    """
    padding = 0
    min_len = 1
    max_len = 200
    batch_size = 50
    stop_words = stop_words

    max_len_actual, doc_cnt, word2id, word2cnt = get_vocab(file, min_len, max_len, stop_words)

    print('Number of documents:', doc_cnt)
    print('Max segment length:', max_len_actual)
    print('Vocabulary size:', len(word2id)+1)

    data = []
    scodes = []
    original = []
    
    train_df = pd.read_csv(file)
    for index, row in train_df.iterrows():        
        scode = row['scode']
        sentence = row['sentence']
        words = line_to_words(sentence, min_len, max_len, stop_words)
        if words is None:
            continue                
        seg_ids = [word2id[word] for word in words]
        seg_ids = [0] * padding + seg_ids + [0] * padding

        data.append(seg_ids)
        scodes.append(str(scode))
        original.append(sentence.encode('utf-8'))
    
    return word2id, data, scodes, original, word2cnt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', help='name of hdf5 train file(YELP/AMAZON/CHIMOBILE/FIQA_post/FIQA_headline)', type=str)
    parser.add_argument('--corpus_type', help='', type=str, default='sent')
    parser.add_argument('--batch_size', help='batch_size of hdf5 train file', type=int, default=50)
    args = parser.parse_args()

    # input file    
    os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.path.abspath(os.path.dirname(os.getcwd()))
    # os.path.abspath(os.path.join(os.getcwd(), "."))
    
    train_name = f'{args.dataset}_{args.corpus_type}'
    args_dataset = os.sep.join([ 'preprocessed', args.dataset ,f"{train_name}_5w.csv",])
    args_w2v  = os.sep.join([ 'preprocessed', args.dataset, f"{train_name}_5w_wv.model"])
    args_name = os.sep.join([ 'preprocessed', args.dataset, f"{train_name}"])    

    print(f"process train_name: {train_name}")   
    print(f"read: {args_dataset}")
    print(f"read: {args_w2v}") 

    stop_words = set(stopwords.words('english'))
    word2id, data, scodes, original, word2cnt = load_data(args_dataset, stop_words = stop_words)
    
    # save word ID (YELP_MATE_word_mapping.txt)
    with open(args_name + '_word_mapping.txt', 'w') as f:
        f.write('<PAD> 0\n')
        for word, idx in sorted(word2id.items(), key=operator.itemgetter(1)):
            f.write("%s %d\n" % (word, idx))
    
    # save count of words (YELP_MATE_word_counts.txt)
    with open(args_name + '_word_counts.txt', 'w') as f:
        for word, count in sorted(word2cnt.items(), key=operator.itemgetter(1), reverse=True):
            f.write("%s %d\n" % (word, count))

    vocab_size = len(word2id) + 1
    w2v = load_vec(args_w2v, word2id)

    embed = np.random.uniform(-0.25, 0.25, (vocab_size, 300)) 
    embed[0] = 0
    for word, vec in w2v.items():
        embed[word2id[word]] = vec

    data, scodes, original = zip(*sorted(
        sample(list(zip(data, scodes, original)), len(data)),
        key=lambda x:len(x[0])))
    
    # save training data of MATE (YELP.hdf5)
    filename = f"{args_name}_train.hdf5"
    with h5py.File(filename, 'w') as f:
        f['w2v'] = np.array(embed)
        # 把資料依據 batch_size 切開
        for i, (segments, codes, segs_o), in enumerate(parallel_chunks(data, scodes, original, args.batch_size)):
            max_len_batch = len(max(segments, key=len))
            batch_id = str(i)
            for j in range(len(segments)):
                segments[j].extend([0] * (max_len_batch - len(segments[j])))
            f['data/' + batch_id] = np.array(segments, dtype=np.int32)
            dt = h5py.special_dtype(vlen=bytes)
            f.create_dataset('scodes/' + batch_id, (len(codes),), dtype="S5", data=codes)            
            f.create_dataset('original/' + batch_id, (len(segs_o),), dtype=dt, data=segs_o)
