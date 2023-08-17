import numpy as np
import pickle as pkl
import scipy.sparse as sp
import re
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec
import nltk
from collections import defaultdict
import argparse
import pandas as pd
import json
import os
import shutil
# from pywsd.utils import lemmatize_sentence

def clean_str(string):
    """
    String cleaning
    """
    string = string.lower()
    string = re.sub(r'[$][A-Za-z][\S]*',"<company>", string)
    string = re.sub(r"\n", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"&#34;", " ", string)
    string = re.sub(r"(http://)?www\.[^ ]+", " _url_ ", string)
    string = re.sub(r"\S*https?:\S*", " _url_ ", string)
    string = re.sub(r"[^a-z0-9$\'_]", " ", string)
    string = re.sub(r"_{2,}", "_", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\$+", " $ ", string)
    string = re.sub(r"rrb", " ", string) # added from prep_hdf5_test.py
    string = re.sub(r"lrb", " ", string)
    string = re.sub(r"rsb", " ", string)
    string = re.sub(r"lsb", " ", string)
    string = re.sub(r"(?<=[a-z])I", " I", string)
    string = re.sub(r"(?<= )[0-9]+(?= )", "NUM", string)
    string = re.sub(r"(?<= )[0-9]+$", "NUM", string)
    string = re.sub(r"^[0-9]+(?= )", "NUM", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--mver', help="version of model", type=str)
    parser.add_argument('--graph', help="Graph Type ('original','DP' or 'DP+')", type=str)
    parser.add_argument('--ASSA_round', help="number of update round of sentiment seed words", type=int)

    parser.add_argument('--train_seed_num', help="", type=str)
    parser.add_argument('--dataset', help="MATE output Dataset name (without extension)", type=str)
    parser.add_argument('--corpus_type', help='doc or sent', type=str)
    args = parser.parse_args()
    
    print("     ===== Round {0}: build graph ====== ".format(args.ASSA_round))    

    # aspects of different dataset
    asp_dict = pd.read_pickle('./data/setting/aspect.pkl') 
    asp_name_list = asp_dict[args.dataset]

    print("Build Graph Round :{0}".format(args.ASSA_round))
    print("args :graph {0}  ASSA_round: {1}".format(args.graph, args.ASSA_round))
    
    # make dir of round
    input1 = os.sep.join([ './','TextGCN', 'data_textgcn',args.mver, f"Round{args.ASSA_round}",""])
    input2 = os.sep.join([ './','model_result', f'trainout_{args.dataset}',f"R{args.ASSA_round}.train_out"])
    input3 = os.sep.join([ './','w2v',f"{args.dataset}_{args.corpus_type}_5w_wv.model"])
    input4 = os.sep.join([ './', 'seed', f'sen_seed_25.txt'])

    if os.path.exists(input1):
        shutil.rmtree(input1)
    os.makedirs(input1)
    print("make dir of {0}".format(input1))

    # 開啟 ASSA 訓練資料分類 (yelp.train_out)
    result_pair = []
    f = open(input2, 'r')
    for line in f:
        sp = line.split('\t')
        res_list = list(map(float, sp[1:len(asp_name_list)+1]))
        result_pair.append({"scode": int(sp[0][2:-1]), 
                            "aspect": res_list.index(max(res_list)), 
                            "text":sp[-1].strip()[2:-1]})
    f.close()
    print("open train out {0}".format(input2))

    # 讀取調整過後的句子
    if args.graph != "original":
        print("read adjust graph:",args.graph)
        # read filtered sentence
        input2 = f"./result_ASSA/adjust/{args.graph}.train_out"        
        f = open(input2, 'r')
        sentence_pair = {}
        for line in f:
            sp = line.split('\t')   
            sentence_pair[int(sp[0][2:-1])] = sp[-1].strip()
        f.close()

        # replace sentence
        result_pair
        for d in result_pair:
            d['text'] = sentence_pair[d['scode']]

    # append text of each aspect
    all_doc = defaultdict()
    for i in result_pair:
        all_doc.setdefault(asp_name_list[i['aspect']], []).append(i['text'])

    # number of doc of each aspect
    for i in asp_name_list:
        print(f"{i}: {len(all_doc[i])}")
    
    nltk.download('averaged_perceptron_tagger')
    stopwords = nltk.corpus.stopwords.words('english')    
    
    # reading word2vec model
    model = Word2Vec.load(input3)
    print("load word2vec model",input3)
    tags = set(['MD', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RP', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS', 'NN', 'NNS'])

    # read sentiment seeds
    with open(input4) as f:
        lines = f.read()    
    seeds = lines.split("|")
    train_pos_word = seeds[0].split(" ")[:5]
    train_neg_word = seeds[1].split(" ")[:5]
    all_train_word = train_pos_word + train_neg_word    

    # for each aspect
    for asp_name in asp_name_list:
        dataset = args.dataset + "_" + asp_name # yelp_AMB ..etc
        print(f"Start aspect: {asp_name}")

        # filter word by pos tag and stopwords
        doc_word_list = [] # list of sentences (each aspect)
        for doc in all_doc[asp_name]:
            # doc = " ".join(lemmatize_sentence(doc))
            word_list = nltk.word_tokenize(clean_str(doc))
            pos_tags = nltk.pos_tag(word_list)
            pos_in = []
            for word,pos in pos_tags:
                    if (pos in tags):
                        pos_in.append(word)
            if pos_in == []:
                continue
            filt = [w.lower() for w in pos_in if w.lower() not in stopwords]
            doc_word_list.append(filt)

        word_embeddings_dim = 300

        # build vocab
        word_freq = {}
        word_set = set()
        for doc_words in doc_word_list: # sentences in list of sentences
            for word in doc_words:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
                if word in all_train_word: # skip if in all_train_word
                    continue
                word_set.add(word)

        vocab = all_train_word + list(word_set) # all words used
        vocab_size = len(vocab)

        # tocheck 這個i好像和實際的差1
        word_doc_list = {} # {"word":[doc_id....]}
        for i in range(len(doc_word_list)): # for each sentences
            doc_words = doc_word_list[i]
            appeared = set() # set of words
            for word in doc_words:
                if word in appeared:
                    continue
                if word in word_doc_list:
                    doc_list = word_doc_list[word]
                    doc_list.append(i)
                    word_doc_list[word] = doc_list
                else:
                    word_doc_list[word] = [i] # 每個字出現在第幾個 doc 裡面
                appeared.add(word)

        word_doc_freq = {} # {"word":frequency...}
        for word, doc_list in word_doc_list.items():
            word_doc_freq[word] = len(doc_list)

        word_id_map = {} # {"word":word_id...}
        for i in range(vocab_size):
            word_id_map[vocab[i]] = i

        vocab_str = '\n'.join(vocab)
        
        # vocabulary of each aspect (all words used)
        f_path = input1 + dataset + '_vocab.txt'
        f = open(f_path, 'w')
        f.write(vocab_str)
        f.close()
        
        # word count of each aspect
        df_word_freq = pd.DataFrame(list(word_freq.items()),columns = ['word','freq'])
        wordF_f = f'{input1}{dataset}_wordF.json'
        with open(wordF_f, 'w') as f:
            json.dump(word_freq, f)

        label_list = ["positive", "negative"]

        # get word embedding of word2vec model
        word_vectors = np.random.uniform(-0.01, 0.01, (vocab_size, word_embeddings_dim)) # shape 9857,300
        no_wv_count = 0
        error_word = []
        for i in range(len(vocab)):
            word = vocab[i]
            try:
                vector = model.wv[word]
            except KeyError:
                no_wv_count+=1
                error_word.append(word)
                pass
            word_vectors[i] = vector
        print(f"No wv count: {no_wv_count}")
        print(f"No wv words: {error_word}")

        import scipy.sparse as sp
        # x: feature vectors of training docs, no initial features
        train_size = len(all_train_word)

        real_train_word_names = [] # ['0\ttrain\tpositive' etc...]
        for word in train_pos_word:
            real_train_word_names.append(str(word_id_map[word]) + "\ttrain\tpositive")
        for word in train_neg_word:
            real_train_word_names.append(str(word_id_map[word]) + "\ttrain\tnegative")

        row_x = []
        col_x = []
        data_x = []
        for i in range(len(all_train_word)):
            for j in range(word_embeddings_dim):
                row_x.append(i)
                col_x.append(j)
                data_x.append(word_vectors.item((word_id_map[all_train_word[i]], j)))

        # shape 10 * 300
        x = sp.csr_matrix((data_x, (row_x, col_x)), 
            shape=(train_size, word_embeddings_dim))
        
        y = [] # shape 10 * 2 (one hot label)
        for i in range(train_size):
            word_meta = real_train_word_names[i]
            temp = word_meta.split('\t')
            label = temp[2]
            one_hot = [0 for l in range(len(label_list))]
            label_index = label_list.index(label)
            one_hot[label_index] = 1
            y.append(one_hot)
        y = np.array(y)

        test_index = []
        train_index = []
        for word in word_id_map:
            if word in all_train_word:
                train_index.append(str(word_id_map[word]))
            else:
                test_index.append(str(word_id_map[word]))
        
        # save train and test word ID
        train_index_str = '\n'.join(train_index) # len 10
        test_index_str = '\n'.join(test_index) # len 9847

        tr_index_f = input1 + dataset + '.train.index'
        f = open(tr_index_f, 'w') # './TextGCN/data_textgcn/Round0/yelp_AMB.train.index' etc
        f.write(train_index_str)
        f.close()

        te_index_f = input1 + dataset + '.test.index'
        f = open(te_index_f, 'w')
        f.write(test_index_str)
        f.close()

        # tx: feature vectors of test docs, no initial features
        # tocheck: why only positive label
        real_test_word_names = [] # word id with positive label
        all_test_word = [] # words
        for word in word_id_map:
            if word in all_train_word:
                continue
            all_test_word.append(word)
            real_test_word_names.append(str(word_id_map[word]) + "\ttest\tpositive")
            
        test_size = len(all_test_word)

        row_tx = []
        col_tx = []
        data_tx = []
        
        for i in range(len(all_test_word)):
            for j in range(word_embeddings_dim):
                row_tx.append(i)
                col_tx.append(j)
                data_tx.append(word_vectors.item((word_id_map[all_test_word[i]], j)))
        # shape 9847 * 300
        tx = sp.csr_matrix((data_tx, (row_tx, col_tx)), 
                            shape=(test_size, word_embeddings_dim))

        ty = [] # shape 9847 * 2
        for i in range(test_size):
            word_meta = real_test_word_names[i]
            temp = word_meta.split('\t')
            label = temp[2]
            one_hot = [0 for l in range(len(label_list))]
            ty.append(one_hot)
        ty = np.array(ty)

        # allx: the the feature vectors of both labeled and unlabeled training instances (a superset of x)
        # unlabeled training instances -> words

        row_allx = []
        col_allx = []
        data_allx = []

        doc_size = len(doc_word_list)
        # 加入訓練字詞的特徵
        for i in range(len(all_train_word)):
            for j in range(word_embeddings_dim):
                row_allx.append(int(i))
                col_allx.append(j)
                data_allx.append(word_vectors.item((word_id_map[all_train_word[i]], j)))
        
        # 加入句子的特徵
        for i in range(doc_size):
            doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
            doc_words = doc_word_list[i]
            doc_len = len(doc_words)
            for word in doc_words:
                try:
                    vector = model.wv[word]
                    doc_vec = doc_vec + np.array(vector)
                except KeyError:
                    pass
            
            for j in range(word_embeddings_dim):
                row_allx.append(int(i + len(all_train_word)))
                col_allx.append(j)
                data_allx.append(doc_vec[j] / doc_len)

        row_allx = np.array(row_allx)
        col_allx = np.array(col_allx)
        data_allx = np.array(data_allx)

        allx = sp.csr_matrix((data_allx, (row_allx, col_allx)), 
            shape=(len(all_train_word) + doc_size, word_embeddings_dim))
            
        # shape: 5251 * 2        
        ally = [] 
        for i in range(len(all_train_word)):
            one_hot = [0 for l in range(len(label_list))]
            for word in train_pos_word:
                if i == word_id_map[word]:
                    label_index = label_list.index("positive")
                    one_hot[label_index] = 1
            for word in train_neg_word:
                if i == word_id_map[word]:
                    label_index = label_list.index("negative")
                    one_hot[label_index] = 1
            ally.append(one_hot)

        for i in range(doc_size):
            one_hot = [0 for l in range(len(label_list))]
            ally.append(one_hot)

        ally = np.array(ally)

        # print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

        # word co-occurence with context windows
        window_size = 10
        windows = []
        for words in doc_word_list:
            length = len(words)
            if length <= window_size:
                windows.append(words)
            else:
                for j in range(length - window_size + 1):
                    window = words[j: j + window_size]
                    windows.append(window)

        word_window_freq = {}
        for window in windows:
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                if window[i] in word_window_freq:
                    word_window_freq[window[i]] += 1
                else:
                    word_window_freq[window[i]] = 1
                appeared.add(window[i])

        word_pair_count = {} # iindex,jindex:count
        for window in windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = word_id_map[word_i]
                    word_j = window[j]
                    word_j_id = word_id_map[word_j]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1

        row = []
        col = []
        weight = []

        # pmi as weights
        num_window = len(windows)

        for key in word_pair_count:
            temp = key.split(',')
            i = int(temp[0])
            j = int(temp[1])
            count = word_pair_count[key]
            word_freq_i = word_window_freq[vocab[i]]
            word_freq_j = word_window_freq[vocab[j]]
            pmi = log((1.0 * count / num_window) /
                    (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
            if pmi <= 0:
                continue
            row.append(i)
            col.append(j)
            weight.append(pmi)

        # doc word frequency
        doc_word_freq = {}
        for doc_id in range(len(doc_word_list)):
            words = doc_word_list[doc_id]
            for word in words:
                word_id = word_id_map[word]
                doc_word_str = str(doc_id) + ',' + str(word_id)
                if doc_word_str in doc_word_freq:
                    doc_word_freq[doc_word_str] += 1
                else:
                    doc_word_freq[doc_word_str] = 1

        # weight of doc and word
        for i in range(len(doc_word_list)):
            words = doc_word_list[i]
            doc_word_set = set()
            for word in words:
                if word in doc_word_set:
                    continue
                j = word_id_map[word]
                key = str(i) + ',' + str(j)
                freq = doc_word_freq[key]
                row.append(train_size + test_size + i)
                col.append(j)
                idf = log(1.0 * len(doc_word_list) /
                        word_doc_freq[vocab[j]])
                weight.append(freq * idf)
                doc_word_set.add(word)

        node_size = train_size + doc_size + test_size
        adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))
        print(adj.shape)
        print()


        # dump objects
        
        f = open("{0}{1}.x".format(input1, dataset), 'wb') # 10 * 300
        pkl.dump(x, f)
        f.close()

        f = open("{0}{1}.y".format(input1, dataset), 'wb') # 10 * 2
        pkl.dump(y, f)
        f.close()

        f = open("{0}{1}.tx".format(input1, dataset), 'wb') # 9847 * 300
        pkl.dump(tx, f)
        f.close()

        f = open("{0}{1}.ty".format(input1, dataset), 'wb') # 9847 * 2
        pkl.dump(ty, f)
        f.close()

        f = open("{0}{1}.allx".format(input1, dataset), 'wb') # 5251 * 300
        pkl.dump(allx, f)
        f.close()

        f = open("{0}{1}.ally".format(input1, dataset), 'wb') # 5251 * 2
        pkl.dump(ally, f)
        f.close()

        f = open("{0}{1}.adj".format(input1, dataset), 'wb') # graph weight
        pkl.dump(adj, f)
        f.close()
