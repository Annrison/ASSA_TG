from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
# import tensorflow.compat.v1 as tf # adjust

from utils import *
from models import GCN, MLP
import os
import sys
import json
import pandas as pd
import pydot
import random

# tf.compat.v1.disable_eager_execution()

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    asp_dict = pd.read_pickle('./data/setting/aspect.pkl') 

    # make seed dir
    # sdir = './model_result/{0}/seed/{1}/{2}'.format(sys.argv[5],sys.argv[2],sys.argv[3]) #original
    sdir = './model_result/{0}'.format(sys.argv[5]) # 0608

    # asp_name_list = asp_dict["YELP"] # for debug
    asp_name_list = asp_dict[sys.argv[2]]

    # read sentiment seeds
    with open(f'./seed/sen_seed_25.txt') as f:
        lines = f.read()    
    seeds = lines.split("|")
    N = 5
    train_pos_word = seeds[0].split(" ")[:N]
    train_neg_word = seeds[1].split(" ")[:N]

    # read aspect seeds
    with open(f'./seed/rest_neu_seeds.txt') as f:
        lines = f.read()    
    train_neu_word = lines.split(" ")

    all_train_word = train_pos_word + train_neg_word + train_neu_word
    
    # Set random seed
    seed = 100 # random.randint(1, 200)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Settings
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dict_vocab_str = ""

    print("     ===== Round {0}: generate seed: ====== ".format(sys.argv[4]))
    print("input args {0} {1} {2} {3}".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
    # rdir = './TextGCN/data/{0}/{1}/{2}/Round{3}/'.format(sys.argv[5], sys.argv[2], sys.argv[3], sys.argv[4])
    rdir = './TextGCN/data/{0}/Round{1}/'.format(sys.argv[5], sys.argv[4])
    
    # for each aspect
    testl = [] 
    all_oup = [] # 存所有的預測結果
    all_train = [] # 存所有的預測結果    
    for asp_name in asp_name_list:
        print(f"Start aspect: {asp_name}")
        dataset = sys.argv[1] + "_" + asp_name # "yelp" + "_" + asp_name # for debug
        
        flags = tf.app.flags
        FLAGS = flags.FLAGS
        flags.DEFINE_string('dataset', dataset, 'Dataset string.')
        flags.DEFINE_string('graph', sys.argv[3], 'Graph Type ("original", "DP" or "DP+").')
        flags.DEFINE_float('ASSA_round', sys.argv[4], 'number of update round of sentiment seed words.')
        flags.DEFINE_string('weight_type', sys.argv[6], 'weight type of sentiment seed words.')
        flags.DEFINE_string('seed_type', sys.argv[7], 'seed type of sentiment seed words.')
        
        flags.DEFINE_string('model', 'gcn', 'Model string.')
        flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
        flags.DEFINE_integer('epochs', 15, 'Number of epochs to train.') # 15
        flags.DEFINE_integer('hidden1', 150, 'Number of units in hidden layer 1.')
        flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
        flags.DEFINE_float('weight_decay', 0,'Weight for L2 loss on embedding matrix.')  # 5e-4
        flags.DEFINE_integer('early_stopping', 10,'Tolerance for early stopping (# of epochs).')
        flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

        # word count of each aspect
        f = open(f'{rdir}{FLAGS.dataset}_wordF.json')
        word_freq = json.load(f)
        df_word_freq = pd.DataFrame(list(word_freq.items()),columns = ['word','freq'])
        freq_threshold = df_word_freq.sort_values('freq',ascending=False).iloc[int(len(word_freq)*0.1),1]
        
        # vocabulary of each aspect
        vocab_list = []
        vocabfname = '{0}{1}_vocab.txt'.format(rdir,FLAGS.dataset)
        print(f"open {vocabfname}")
        f = open(vocabfname, 'r')
        lines = f.readlines()
        for line in lines:
            vocab_list.append(line.replace("\n", ""))
        f.close()

        data_path = rdir + FLAGS.dataset
        print("data_path",data_path)
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(data_path)
        features = sp.identity(features.shape[0])  # featureless

        # Some preprocessing
        features = preprocess_features(features)
        if FLAGS.model == 'gcn':
            support = [preprocess_adj(adj)]
            num_supports = 1
            model_func = GCN
        elif FLAGS.model == 'gcn_cheby':
            support = chebyshev_polynomials(adj, FLAGS.max_degree)
            num_supports = 1 + FLAGS.max_degree
            model_func = GCN
        elif FLAGS.model == 'dense':
            support = [preprocess_adj(adj)]  # Not used
            num_supports = 1
            model_func = MLP
        else:
            raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

        # Define placeholders
        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            # helper variable for sparse dropout
            'num_features_nonzero': tf.placeholder(tf.int32)
        }

        # Create model
        model = model_func(placeholders, input_dim=features[2][1], logging=True)

        # Initialize session
        session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        sess = tf.Session(config=session_conf)

        # Define model evaluation function
        def evaluate(features, support, labels, mask, placeholders):
            t_test = time.time()
            feed_dict_val = construct_feed_dict(
                features, support, labels, mask, placeholders)
            outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels, model.predict()], feed_dict=feed_dict_val)
            return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test), outs_val[4]

        # Init variables
        sess.run(tf.global_variables_initializer())
        cost_val = []

        # Train model
        for epoch in range(FLAGS.epochs):

            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy,
                            model.layers[0].embedding], feed_dict=feed_dict)
            
            print("Epoch:", '%04d' % (epoch + 1), 
                "train_loss=", "{:.5f}".format(outs[1]),
                "train_acc=", "{:.5f}".format( outs[2]), 
                "time=", "{:.5f}".format(time.time() - t))

            all_train.append([asp_name, (epoch + 1), outs[1],outs[2],(time.time() - t)])

        print("Optimization Finished!")

        # Testing
        test_cost, test_acc, pred, labels, test_duration, oup = evaluate(
            features, support, y_test, test_mask, placeholders)

        predict_pos_word = dict()
        predict_neg_word = dict()

        # s = len(test_mask) - test_size
        # for i in range(0, len(test_mask)):
        #     if test_mask[i]:
        #         j = (i - s) + 10
        #         pos_prob = oup[i][0]
        #         neg_prob = oup[i][1]
        #         word_str = vocab_list[j]
        #         if (word_freq[word_str] >= freq_threshold) and (len(word_str)>2) and (":" not in word_str) and (word_str not in train_neu_word):                    
        #             all_oup.append([asp_name,word_str,pos_prob,neg_prob])                    
        #             # # not filter by threshold
        #             predict_pos_word[word_str] = pos_prob
        #             predict_neg_word[word_str] = neg_prob

        # filter words
        # (1) > freq_threshold
        # (2) len(word) > 2
        # shape of oup: 15098, 2

        # pthres = 0.6 ; nthres = 0.65
        pthres = 0.5 ; nthres = 0.5
        random_seed_voc = []
        
        s = len(oup) - test_size
        for i in range(10, len(oup)):
            if i < len(vocab_list):
                try:                    
                    word_str = vocab_list[i]
                    if (word_freq[word_str] >= freq_threshold) and (len(word_str)>2) and (":" not in word_str) and (word_str not in all_train_word):                    
                        all_oup.append([asp_name,word_str,oup[i][0],oup[i][1]])
                        random_seed_voc.append(word_str)
                        
                        # # not filter by threshold
                        predict_pos_word[word_str] = oup[i][0]
                        predict_neg_word[word_str] = oup[i][1]

                        # filter by threshold (pos: 0.6, neg: 0.65)
                        if (oup[i][0] > pthres) & (oup[i][1] < nthres): 
                            predict_pos_word[vocab_list[i]] = oup[i][0]

                        elif (oup[i][1] > nthres) & (oup[i][0] < pthres): # 0.65
                            predict_neg_word[vocab_list[i]] = oup[i][1]
                except:
                    print("vocab_list[i]",vocab_list[i])
        
        predict_pos_word = sorted(predict_pos_word.items(), key=lambda d: d[1], reverse=True) # 48
        predict_neg_word = sorted(predict_neg_word.items(), key=lambda d: d[1], reverse=True) # 787

        # length: to pick top n(length) words
        length = 20
        length_min = min(len(predict_pos_word), len(predict_neg_word))        
        if length > length_min:
            length = length_min
            print("length_min",length_min)

        general_seed = False
        weight = 0.1
        
        # general + textGCN seed
        if FLAGS.seed_type == "add":
            general_seed = True
            weight = 0.05        

        # random_seed = random.sample(random_seed_voc, 10)

        # positive words
        if general_seed:
            for word in train_pos_word:
                dict_vocab_str += word+ ":" + str(weight) + " "
        if FLAGS.weight_type == "fix":
            # # official
            for i in range(length):
                dict_vocab_str += predict_pos_word[i][0]+ ":" + str(weight) + " " # 測權重，記得改過來
            # # length test
            # for i in range(int(sys.argv[4])+1):
            #     dict_vocab_str += predict_pos_word[i][0]+":0.1 "
        else:
            # trained weight
            for i in range(length):
                dict_vocab_str += predict_pos_word[i][0]+":" + str(predict_pos_word[i][1]) + " "
            # random test
            # for word in random_seed[-length:]:
            #     dict_vocab_str += word+":0.5 "
        dict_vocab_str += "| "

        # negative words
        if general_seed:
            for word in train_neg_word:
                dict_vocab_str += word+ ":" + str(weight) + " "
        if FLAGS.weight_type == "fix":             
            # # official
            for i in range(length):
                dict_vocab_str += predict_neg_word[i][0]+ ":" + str(weight) + " " # 測權重，記得改過來
            # # length test
            # for i in range(int(sys.argv[4])+1):
            #     dict_vocab_str += predict_neg_word[i][0]+":0.1 "
        else:
            # trained weight
            for i in range(length):
                dict_vocab_str += predict_neg_word[i][0]+":" + str(predict_neg_word[i][1]) + " "            
            # # random test
            # for word in random_seed[:length]:
            #     dict_vocab_str += word+":0.5 "

        dict_vocab_str += "\n"

        del_all_flags(flags.FLAGS)

    # all aspect in a file (per domain) # todo    
    seed_path = "{0}/GCN_seed_R{1}.txt".format(sdir,int(sys.argv[4])+1)  
    f = open(seed_path, 'w')
    f.write(dict_vocab_str)
    f.close()
    print("save seed to: {0}".format(seed_path))  

    df = pd.DataFrame(all_train)
    df.to_csv("{0}/R{1}_train.csv".format(sdir,int(sys.argv[4])+1), encoding='utf-8', index=False)

    df = pd.DataFrame(all_oup)
    df.to_csv("{0}/R{1}_oup.csv".format(sdir,int(sys.argv[4])+1), encoding='utf-8', index=False)

    # test cmd 
    # python ./TextGCN/train.py yelp YELP yelp 0 0324_8 fix