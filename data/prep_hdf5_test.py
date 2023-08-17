#!/usr/bin/env python
import pandas as pd
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
from nltk import tokenize
# from pywsd.utils import lemmatize_sentence
# nltk.download('omw-1.4')
# nltk.download('wordnet')

def equal_length_chunks(l1, l2, l3, l4, l5):
    """
    Splits list of instances into batches, so that segments within batches
    are of equal length
    """
    prev = -1
    for i in range(len(l1)):
        if len(l1[i]) != prev:
            if prev != -1:
                yield l1[start:end+1], l2[start:end+1], l3[start:end+1], l4[start:end+1], l5[start:end+1]
            start = i
            end = i
        else:
            end += 1
        prev = len(l1[i])
    yield l1[start:end+1], l2[start:end+1], l3[start:end+1], l4[start:end+1], l5[start:end+1]

def line_to_words(line, min_len, max_len, stop_words=None):
    """
    Reads a line of text (sentence) and returns a list of tokenized EDUs
    """

    # word_list = word_tokenize(line)    
    word_list = line.split(' ')    

    if stop_words is not None:
        words = [word for word in word_list if word not in stop_words]

    # discard short segments
    if len(words) < min_len:
        return None
    
    # truncate long ones
    if len(words) > max_len:
        words = words[:max_len]

    return words

def load_data(file, args, word2id, num_asp, num_sen):
    """
    Loads test data
    """
    padding = args.padding
    batch_size = args.batch_size
    stop_words = args.stop_words

    data = []
    labels_asp = []
    labels_sen = []
    scodes = []
    original = []

    doc_cnt = 0
    
    test_df = pd.read_csv(file)
    for index, row in test_df.iterrows():        
        idx = row['scode']
        text = row['sentence']
        label_asp = row['aspect']
        label_sen = row['sentiment']

        words = line_to_words(text, 1, 10000, stop_words=stop_words)
        if words is None:
            continue

        seg_ids = [word2id[word] if word in word2id else 1 for word in words]
        seg_ids = [0] * padding + seg_ids + [0] * padding

        seg_asp = [0] * num_asp
        seg_asp[int(label_asp)] = 1
        
        seg_sen = [0] * num_sen
        seg_sen[int(label_sen)] = 1

        data.append(seg_ids)
        labels_asp.append(seg_asp)
        labels_sen.append(seg_sen)
        scodes.append(str(idx))
        original.append(text.encode("utf8"))
        doc_cnt += 1

    print('Number of documents:', doc_cnt)
    print('Number of aspects:', num_asp)
    print('Vocabulary size:', len(word2id))

    return data, labels_asp, labels_sen, scodes, original

def main():
    parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', help='name of dataset (YELP/AMAZON/FIQA_POST/FIQA_HEADLINE)', type=str)
    parser.add_argument('--corpus_type', help='', type=str, default='sent')
    parser.add_argument('--sentiments', help='number of sentiment', type=int, default=2)
    parser.add_argument('--batch_size', help='maximum number of segments per batch (default: 50)', type=int, default=50)
    parser.add_argument('--padding', help='padding around each sentence (default: 0)', type=int, default=0)
    parser.add_argument('--stopfile', help='Stop-word file (default: nltk english stop words)', type=str, default='')
    parser.add_argument('--seed', help='random seed (default: 1)', type=int, default=1)

    args = parser.parse_args()
    
    train_name = f'{args.dataset}_{args.corpus_type}'
    
    args_data = f'test/test_{train_name}.csv'
    args_vocab = f'preprocessed/{args.dataset}/{train_name}_word_mapping.txt'
    asp_dict = pd.read_pickle('setting/aspect.pkl')    
    args_aspects = len(asp_dict[args.dataset])

    print(f"process dataset: {train_name}")

    if args.stopfile == 'no':
        args.stop_words = None
    elif args.stopfile != '':
        stop_words = set()
        fstop = open(args.stopfile, 'r')
        for line in fstop:
            stop_words.add(line.strip())
        fstop.close()
        args.stop_words = stop_words
    else:
        args.stop_words = set(stopwords.words('english'))

    word2id = {}    
    fvoc = open(args_vocab, 'r')
    for line in fvoc:
        word, id = line.split()
        word2id[word] = int(id)
    fvoc.close()

    data, label_asp, label_sen, scodes, original = load_data(args_data, args, word2id, args_aspects, args.sentiments)

    seed(args.seed)
    data, label_asp, label_sen, scodes, original = zip(*sorted(
        sample(list(zip(data, label_asp, label_sen, scodes, original)), len(data)),
        key=lambda x:len(x[0])))

    # output testing data (YELP_TEST.hdf5)
    filename = f"./preprocessed/{args.dataset}/{train_name}_test" + '.hdf5'
    with h5py.File(filename, 'w') as f:
        for i, (segments, lbls, lbls_sen, codes, segs_o), in enumerate(equal_length_chunks(data, label_asp, label_sen, scodes, original)):
            max_len_batch = len(max(segments, key=len))
            batch_id = str(i)
            for j in range(len(segments)):
                segments[j].extend([0] * (max_len_batch - len(segments[j])))
            f['data/' + batch_id] = np.array(segments, dtype=np.int32)
            f['labels/' + batch_id] = np.array(lbls, dtype=np.int32)
            f['labels_sen/' + batch_id] = np.array(lbls_sen, dtype=np.int32)
            dt = h5py.special_dtype(vlen=bytes)
            f.create_dataset('scodes/' + batch_id, (len(codes),), dtype="S4", data=codes)            
            f.create_dataset('original/' + batch_id, (len(segs_o),), dtype=dt, data=segs_o)

if __name__ == '__main__':
    main()