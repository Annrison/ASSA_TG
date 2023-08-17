from __future__ import division
from __future__ import print_function

import time
# import tensorflow as tf
import tensorflow.compat.v1 as tf

from utils import *
from models import GCN, MLP
import os
import sys
import json
import pandas as pd

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    asp_dict = {
        "REST": ["AMB", "DRI", "FOD", "LOC", "SRV"],
        "LAPTOP": ["SUPPORT", "OS", "DISPLAY", "BATTERY", "COMPANY", "MOUSE", "SOFTWARE", "KEYBOARD"],
        "FIQA" : ["STOCK","CORPORATE","MARKET","ECONOMY"]
    }

    for_test = False # False True
    if for_test:
        sys_argv1 = "yelp"
        sys_argv2 = "REST"
        sys_argv3 = "yelp"
        sys_argv4 = 0

    asp_name_list = asp_dict[sys.argv[2]]
    if for_test:
        asp_name_list = asp_dict[sys_argv2]

    train_pos_word = ["good", "great", "nice", "best", "amazing"]
    train_neg_word = ["gross", "bad", "terrible", "hate", "disappointed"]
    
    # corpus dir
    rdir = './TextGCN/data/{0}/{1}/Round{2}/'.format(sys.argv[2], sys.argv[3], sys.argv[4])
    if for_test:
        rdir = './TextGCN/data/Round{0}/'.format(sys_argv4)

    # Set random seed
    seed = 100 # random.randint(1, 200)
    np.random.seed(seed)
#     tf.set_random_seed(seed)
    tf.random.set_random_seed(seed) # adjust

    # Settings
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # for each aspect 
    dict_vocab_str = ""
    for asp_name in asp_name_list:
        print(f"Start aspect: {asp_name}")
        dataset = sys.argv[1] + "_" + asp_name
        if for_test:
            dataset = sys_argv1 + "_" + asp_name

        flags = tf.app.flags
        FLAGS = flags.FLAGS
        flags.DEFINE_string('dataset', dataset, 'Dataset string.')
        flags.DEFINE_string('model', 'gcn', 'Model string.')
        flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
        flags.DEFINE_integer('epochs', 15, 'Number of epochs to train.')
        flags.DEFINE_integer('hidden1', 150, 'Number of units in hidden layer 1.')
        flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
        flags.DEFINE_float('weight_decay', 0,'Weight for L2 loss on embedding matrix.')  # 5e-4
        flags.DEFINE_integer('early_stopping', 10,'Tolerance for early stopping (# of epochs).')
        flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

        # Load data
        data_path = rdir + FLAGS.dataset
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
        tf.compat.v1.disable_eager_execution() # adjust
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

        # Train model
        cost_val = []
        for epoch in range(FLAGS.epochs):

            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy,
                            model.layers[0].embedding], feed_dict=feed_dict)

            
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                "train_acc=", "{:.5f}".format( outs[2]), "time=", "{:.5f}".format(time.time() - t))

        print("Optimization Finished!")

        # Testing
        test_cost, test_acc, pred, labels, test_duration, oup = evaluate(
            features, support, y_test, test_mask, placeholders)

        test_pred = [] # [0,1,0....]
        test_labels = [] # [0,1,1....]
        # print(len(test_mask))
        for i in range(len(test_mask)):
            if test_mask[i]:
                test_pred.append(pred[i])
                test_labels.append(labels[i])

        # read word frenquency
        f = open(f'{data_path}_wordF.json')
        word_freq = json.load(f) # {"think":2,"direct":4....}
        df_word_freq = pd.DataFrame(list(word_freq.items()),columns = ['word','freq'])
        freq_threshold = 0 # df_word_freq.sort_values('freq',ascending=False).iloc[int(len(word_freq)*0.5),1]

        # read word vocabulary
        vocab_list = []
        f = open(f'{data_path}_vocab.txt', 'r')
        lines = f.readlines()
        for line in lines:
            vocab_list.append(line.replace("\n", ""))
        f.close()

        predict_pos_word = dict()
        predict_neg_word = dict()

        OOV = []
        for i in range(0, len(oup)):
            if i < len(vocab_list):
                if vocab_list[i] in word_freq:
                    if word_freq[vocab_list[i]] >= freq_threshold and len(vocab_list[i])>2 and ":" not in vocab_list[i]:
                        if oup[i][0] > 0.5:
                            predict_pos_word[vocab_list[i]] = oup[i][0]-0.5
                        elif oup[i][1] > 0.5:
                            predict_neg_word[vocab_list[i]] = -(oup[i][1]-0.5)
                    else:
                        # 應該發生在當訓練資料的情緒種子沒有在那個aspect文集中的情況
                        # 檢查 build_graph.py line 114
                        OOV.append(vocab_list[i])
        print("NOTICE!!")
        print("voc not in word_freq",OOV)

        predict_pos_word = sorted(predict_pos_word.items(), key=lambda d: d[1], reverse=True) # [('amazing', 0.49676263332366943), ....
        predict_neg_word = sorted(predict_neg_word.items(), key=lambda d: d[1], reverse=True)

        # length = min(len(predict_pos_word), len(predict_neg_word))

        for i in range(len(predict_pos_word)):
            dict_vocab_str += f"{predict_pos_word[i][0]}:{predict_pos_word[i][1]:.2f} " # word:polarity
        dict_vocab_str += "| "
        for i in range(len(predict_neg_word)):
            dict_vocab_str += f"{predict_neg_word[i][0]}:{predict_neg_word[i][1]:.2f} "
        dict_vocab_str += "\n"

        del_all_flags(flags.FLAGS)

    # 每個aspect的字是分開來存的
    f = open(f'./seed/{sys.argv[2]}/{sys.argv[3]}/GCN_all_R{sys.argv[4]}.txt', 'w')
    if for_test:
        f = open(f'./seed/{sys_argv2}_GCN_all_R{sys_argv4}.txt', 'w')
    f.write(dict_vocab_str)
    f.close()