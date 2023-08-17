#!/usr/bin/env python

import sys
import h5py
import argparse
from time import time

import numpy as np
from numpy.random import permutation, seed
from scipy.cluster.vq import kmeans

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_uniform
from subprocess import call
import csv
import os
import shutil

from loss import TripletMarginCosineLoss, OrthogonalityLoss
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def count_acc(fname, asp_num, sen_num):
    print("count_acc_fname",fname)
    result_pair = []
    f = open(fname, 'r')
    for line in f:
        sp = line.split('\t')
        asp_res_list = list(map(float, sp[1:asp_num+1]))
        sen_res_list = list(map(float, sp[asp_num+1:asp_num+sen_num+1]))
        result_pair.append({"scode": sp[0][2:-1], "asp":asp_res_list.index(max(asp_res_list)), "sen":sen_res_list.index(max(sen_res_list))})
    f.close()
    y_test_asp = [] # aspect true label
    y_pred_asp = [] # aspect predict label
    y_test_sen = [] # sentiment true label
    y_pred_sen = [] # sentiment predict label
    for i in result_pair:
        y_test_asp.append(str(i['asp']))
        y_pred_asp.append(id_asp_pair[i['scode']])
        y_test_sen.append(str(i['sen']))
        y_pred_sen.append(id_sen_pair[i['scode']])
    a_acc, a_pre, a_rec, a_f1 = count_aprf(y_test_asp, y_pred_asp)
    s_acc, s_pre, s_rec, s_f1 = count_aprf(y_test_sen, y_pred_sen)
    print(f"-  ASPECT   - Accuracy: {a_acc:.5f} Precision: {a_pre:.5f} Recall: {a_rec:.5f} F1: {a_f1:.5f}")
    print(f"- SENTIMENT - Accuracy: {s_acc:.5f} Precision: {s_pre:.5f} Recall: {s_rec:.5f} F1: {s_f1:.5f}")
    return a_acc, a_pre, a_rec, a_f1, s_acc, s_pre, s_rec, s_f1

def count_aprf(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average = 'macro')
    rec = recall_score(y_test, y_pred, average = 'macro')
    f1 = f1_score(y_test, y_pred, average = 'macro')
    return acc,pre,rec,f1

class AttentionEncoder(nn.Module):
    """Segment encoder that produces segment vectors as the weighted
    average of word embeddings.
    """
    def __init__(self, vocab_size, emb_size, bias=True, M=None, b=None):
        """Initializes the encoder using a [vocab_size x emb_size] embedding
        matrix. The encoder learns a matrix M, which may be initialized
        explicitely or randomly.

        Parameters:
            vocab_size (int): the vocabulary size
            emb_size (int): dimensionality of embeddings
            bias (bool): whether or not to use a bias vector
            M (matrix): the attention matrix (None for random)
            b (vector): the attention bias vector (None for random)
        """
        super(AttentionEncoder, self).__init__()
        self.lookup = nn.Embedding(vocab_size, emb_size)
        self.M = nn.Parameter(torch.Tensor(emb_size, emb_size))
        if M is None:
            xavier_uniform(self.M.data)
        else:
            self.M.data.copy_(M)
        if bias:
            self.b = nn.Parameter(torch.Tensor(1))
            if b is None:
                self.b.data.zero_()
            else:
                self.b.data.copy_(b)
        else:
            self.b = None

    def forward(self, inputs):
        """Forwards an input batch through the encoder"""
        x_wrd = self.lookup(inputs)
        x_avg = x_wrd.mean(dim=1)

        x = x_wrd.matmul(self.M)
        x = x.matmul(x_avg.unsqueeze(1).transpose(1,2))
        if self.b is not None:
            x += self.b

        x = F.tanh(x)
        a = F.softmax(x, dim=1)

        z = a.transpose(1,2).matmul(x_wrd)
        z = z.squeeze()
        if z.dim() == 1:
            return z.unsqueeze(0)
        return z

    def set_word_embeddings(self, embeddings, fix_w_emb=True):
        """Initialized word embeddings dictionary and defines if it is trainable"""
        self.lookup.weight.data.copy_(embeddings)
        self.lookup.weight.requires_grad = not fix_w_emb

class AspectAutoencoder(nn.Module):
    """The aspect autoencoder class that defines our Multitask Aspect Extractor,
    but also implements the Aspect-Based Autoencoder (ABAE), if the aspect matrix
    not initialized using seed words
    """
    def __init__(self, vocab_size, emb_size, num_aspects=10, neg_samples=10,
            w_emb=None, a_emb=None, recon_method='centr', seed_w=None, num_seeds=None,
            attention=False, bias=True, M=None, b=None, fix_w_emb=True, fix_a_emb=False):
        """Initializes the autoencoder instance.

        Parameters:
            vocab_size (int): the vocabulary size
            emb_size (int): the embedding dimensionality
            num_aspects (int): the number of aspects
            neg_samples (int): the number of negative examples to use for the 
                               max-margin loss
            w_emb (matrix): a pre-trained embeddings matrix (None for random)
            a_emb (matrix): a pre-trained aspect matrix (None for random)
            recon_method (str): the segment reconstruction policy
                                - 'centr': uses centroid of seed words or single embeddings (ABAE)
                                - 'init': uses manually initialized seed weights
                                - 'fix': uses manually initialized seed weights, fixed during training
                                - 'cos': uses dynamic seed weights, obtained from cosine distance
            seed_w (matrix): seed weight matrix (for 'init' and 'fix')
            num_seeds (int): number of seed words
            attention (bool): use attention or not
            bias (bool): use bias vector for attention encoder
            M (matrix): matrix for attention encoder (optional)
            b (vector): bias vector for attention encoder (optional)
            fix_w_emb (bool): fix word embeddings throughout trainign
            fix_a_emb (bool): fix aspect embeddings throughout trainign
        """
        super(AspectAutoencoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.recon_method = recon_method
        self.num_seeds = num_seeds
        self.attention = attention
        self.bias = bias
        self.num_aspects = num_aspects
        self.neg_samples = neg_samples

        if not attention:
            self.seg_encoder = nn.EmbeddingBag(vocab_size, emb_size)
        else:
            self.seg_encoder = AttentionEncoder(vocab_size, emb_size, bias, M, b)
        self.encoder_hook = self.seg_encoder.register_forward_hook(self.set_targets)

        if w_emb is None:
            xavier_uniform(self.seg_encoder.weight.data)
        else:
            assert w_emb.size() == (vocab_size, emb_size), "Word embedding matrix has incorrect size"
            if not attention:
                self.seg_encoder.weight.data.copy_(w_emb)
                self.seg_encoder.weight.requires_grad = not fix_w_emb
            else:
                self.seg_encoder.set_word_embeddings(w_emb, fix_w_emb)

        if a_emb is None:
            self.a_emb = nn.Parameter(torch.Tensor(num_aspects, emb_size))
            xavier_uniform(self.a_emb.data)
        else:
            assert a_emb.size()[0] == num_aspects and a_emb.size()[-1] == emb_size, "Aspect embedding matrix has incorrect size"
            self.a_emb = nn.Parameter(torch.Tensor(a_emb.size()))
            self.a_emb.data.copy_(a_emb)
            self.a_emb.requires_grad = not fix_a_emb
        
        if recon_method == 'fix':
            self.seed_w = nn.Parameter(torch.Tensor(seed_w.size()))
            self.seed_w.data.copy_(seed_w)
            self.seed_w.requires_grad = False
        elif recon_method == 'init':
            self.seed_w = nn.Parameter(torch.Tensor(seed_w.size()))
            self.seed_w.data.copy_(seed_w)
        else:
            self.seed_w = None

        self.lin = nn.Linear(emb_size, num_aspects)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, batch_num=None):
        if self.training:
            # mask used for randomly selected negative examples
            self.cur_mask = self._create_neg_mask(inputs.size(0))
        if not self.attention:
            offsets = Variable(torch.arange(0, inputs.numel(), inputs.size(1), out=inputs.data.new().long()))
            enc = self.seg_encoder(inputs.view(-1), offsets)
        else:
            enc = self.seg_encoder(inputs)

        x = self.lin(enc)
        a_probs = self.softmax(x)
        
        if self.recon_method == 'centr':
            r = a_probs.matmul(self.a_emb)
        elif self.recon_method == 'fix':
            a_emb_w = self.a_emb.mul(self.seed_w.view(self.num_aspects, self.num_seeds, 1))
            r = a_probs.view(-1, self.num_aspects, 1, 1).mul(a_emb_w).sum(dim=2).sum(dim=1)
        elif self.recon_method == 'init':
            seed_w_norm = F.softmax(self.seed_w, dim=1)
            a_emb_w = self.a_emb.mul(seed_w_norm.view(self.num_aspects, self.num_seeds, 1))
            r = a_probs.view(-1, self.num_aspects, 1, 1).mul(a_emb_w).sum(dim=2).sum(dim=1)
        elif self.recon_method == 'cos':
            sim = F.cosine_similarity(enc.unsqueeze(1),
                    self.a_emb.view(1, self.num_aspects*self.num_seeds, self.emb_size),
                    dim=2).view(-1, self.num_aspects, self.num_seeds)
            self.seed_w = F.softmax(sim, dim=2)
            a_emb_w = self.a_emb.mul(self.seed_w.view(-1, self.num_aspects, self.num_seeds, 1))
            r = a_probs.view(-1, self.num_aspects, 1, 1).mul(a_emb_w).sum(dim=2).sum(dim=1)

        return r, a_probs

    def _create_neg_mask(self, batch_size):
        """Creates a mask for randomly selecting negative samples"""
        multi_weights = torch.ones(batch_size, batch_size) - torch.eye(batch_size)
        neg = min(batch_size - 1, self.neg_samples)

        mask = torch.multinomial(multi_weights, neg)
        mask = mask.unsqueeze(2).expand(batch_size, neg, self.emb_size)
        mask = Variable(mask, requires_grad=False)
        return mask

    def set_targets(self, module, input, output):
        """Sets positive and negative samples"""
        assert self.cur_mask is not None, 'Tried to set targets without a mask'
        batch_size = output.size(0)

        if torch.cuda.is_available():
            mask = self.cur_mask.cuda()
        else:
            mask = self.cur_mask

        self.negative = Variable(output.data).expand(batch_size, batch_size, self.emb_size).gather(1, mask)
        self.positive = Variable(output.data)
        self.cur_mask = None

    def get_targets(self):
        assert self.positive is not None, 'Positive targets not set; needs a forward pass first'
        assert self.negative is not None, 'Negative targets not set; needs a forward pass first'
        return self.positive, self.negative

    def get_aspects(self):
        if self.a_emb.dim() == 2:
            return self.a_emb
        else:
            return self.a_emb.mean(dim=1)

    def train(self, mode=True):
        super(AspectAutoencoder,  self).train(mode)
        if self.encoder_hook is None:
            self.encoder_hook = self.seg_encoder.register_forward_hook(self.set_targets)
        return self

    def eval(self):
        super(AspectAutoencoder, self).eval()
        if self.encoder_hook is not None:
            self.encoder_hook.remove()
            self.encoder_hook = None
        return self

class SentimentAutoencoder(nn.Module):
    """The sentiment autoencoder class that defines our Multitask Aspect Extractor,
    but also implements the Aspect-Based Autoencoder (ABAE), if the aspect matrix
    not initialized using seed words
    """
    def __init__(self, vocab_size, emb_size, num_aspects=5, num_sens=2, neg_samples=10,
            w_emb=None, s_emb=None, recon_method='centr', seed_w=None, num_seeds=None,
            attention=False, bias=True, M=None, b=None, fix_w_emb=True, fix_s_emb=False):
        """Initializes the autoencoder instance.

        Parameters:
            vocab_size (int): the vocabulary size
            emb_size (int): the embedding dimensionality
            num_sens (int): the number of sentiments
            neg_samples (int): the number of negative examples to use for the 
                               max-margin loss
            w_emb (matrix): a pre-trained embeddings matrix (None for random)
            s_emb (matrix): a pre-trained sentiment matrix (None for random)
            recon_method (str): the segment reconstruction policy
                                - 'centr': uses centroid of seed words or single embeddings (ABAE)
                                - 'init': uses manually initialized seed weights
                                - 'fix': uses manually initialized seed weights, fixed during training
                                - 'cos': uses dynamic seed weights, obtained from cosine distance
            seed_w (matrix): seed weight matrix (for 'init' and 'fix')
            num_seeds (int): number of seed words
            attention (bool): use attention or not
            bias (bool): use bias vector for attention encoder
            M (matrix): matrix for attention encoder (optional)
            b (vector): bias vector for attention encoder (optional)
            fix_w_emb (bool): fix word embeddings throughout trainign
            fix_s_emb (bool): fix sentiment embeddings throughout trainign
        """
        super(SentimentAutoencoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.recon_method = recon_method
        self.num_seeds = num_seeds
        self.attention = attention
        self.bias = bias
        self.num_asps = num_aspects
        self.num_sens = num_sens
        self.neg_samples = neg_samples

        if not attention:
            self.seg_encoder = nn.EmbeddingBag(vocab_size, emb_size)
        else:
            self.seg_encoder = AttentionEncoder(vocab_size, emb_size, bias, M, b)
        self.encoder_hook = self.seg_encoder.register_forward_hook(self.set_targets)

        if w_emb is None:
            xavier_uniform(self.seg_encoder.weight.data)
        else:
            assert w_emb.size() == (vocab_size, emb_size), "Word embedding matrix has incorrect size"
            if not attention:
                self.seg_encoder.weight.data.copy_(w_emb)
                self.seg_encoder.weight.requires_grad = not fix_w_emb
            else:
                self.seg_encoder.set_word_embeddings(w_emb, fix_w_emb)

        if s_emb is None:
            self.s_emb = nn.Parameter(torch.Tensor(num_sens, emb_size))
            xavier_uniform(self.s_emb.data)
        else:
            # assert s_emb.size()[0] == num_sens and s_emb.size()[-1] == emb_size, "Sentiment embedding matrix has incorrect size"
            self.s_emb = nn.Parameter(torch.Tensor(s_emb.size()))
            self.s_emb.data.copy_(s_emb)
            self.s_emb.requires_grad = not fix_s_emb
        # print(seed_w.size())
        if recon_method == 'fix':
            self.seed_w = nn.Parameter(torch.Tensor(seed_w.size()))
            self.seed_w.data.copy_(seed_w)
            self.seed_w.requires_grad = False
        elif recon_method == 'init':
            self.seed_w = nn.Parameter(torch.Tensor(seed_w.size()))
            self.seed_w.data.copy_(seed_w)
        else:
            self.seed_w = None

        self.lin = nn.Linear(emb_size, num_sens)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, asp_probs, batch_num=None):
        if self.training:
            # mask used for randomly selected negative examples
            self.cur_mask = self._create_neg_mask(inputs.size(0))
        if not self.attention:
            offsets = Variable(torch.arange(0, inputs.numel(), inputs.size(1), out=inputs.data.new().long()))
            enc = self.seg_encoder(inputs.view(-1), offsets)
        else:
            enc = self.seg_encoder(inputs)

        x = self.lin(enc)
        a_probs = self.softmax(x)
        # print(a_probs[0])
        # print(asp_probs[0])
        as_probs = torch.bmm(asp_probs.unsqueeze(2),a_probs.unsqueeze(1)).view(-1, self.num_asps * self.num_sens)
        # print(as_probs[0])
        # print(as_probs.shape)
        # print(self.s_emb.shape)

        if self.recon_method == 'centr':
            r = a_probs.matmul(self.s_emb)
        elif self.recon_method == 'fix':
            s_emb_w = self.s_emb.mul(self.seed_w.view(self.num_asps, self.num_sens, self.num_seeds, 1))
            # print(self.seed_w.shape)
            # print(s_emb_w.shape)
            r = as_probs.view(-1, self.num_asps, self.num_sens, 1, 1).mul(s_emb_w).sum(dim=3).sum(dim=2).sum(dim=1)
            # print(r.shape)
        elif self.recon_method == 'init':
            seed_w_norm = F.softmax(self.seed_w, dim=2)
            s_emb_w = self.s_emb.mul(seed_w_norm.view(self.num_asps, self.num_sens, self.num_seeds, 1))
            # r = a_probs.view(-1, self.num_sens, 1, 1).mul(s_emb_w).sum(dim=2).sum(dim=1)
            r = as_probs.view(-1, self.num_asps, self.num_sens, 1, 1).mul(s_emb_w).sum(dim=3).sum(dim=2).sum(dim=1)
        elif self.recon_method == 'cos':
            sim = F.cosine_similarity(enc.unsqueeze(1),
                    self.s_emb.view(1, self.num_sens*self.num_seeds, self.emb_size),
                    dim=2).view(-1, self.num_sens, self.num_seeds)
            self.seed_w = F.softmax(sim, dim=2)
            s_emb_w = self.a_emb.mul(self.seed_w.view(-1, self.num_sens, self.num_seeds, 1))
            r = a_probs.view(-1, self.num_sens, 1, 1).mul(s_emb_w).sum(dim=2).sum(dim=1)

        return r, a_probs

    def _create_neg_mask(self, batch_size):
        """Creates a mask for randomly selecting negative samples"""
        multi_weights = torch.ones(batch_size, batch_size) - torch.eye(batch_size)
        neg = min(batch_size - 1, self.neg_samples)

        mask = torch.multinomial(multi_weights, neg)
        mask = mask.unsqueeze(2).expand(batch_size, neg, self.emb_size)
        mask = Variable(mask, requires_grad=False)
        return mask

    def set_targets(self, module, input, output):
        """Sets positive and negative samples"""
        assert self.cur_mask is not None, 'Tried to set targets without a mask'
        batch_size = output.size(0)

        if torch.cuda.is_available():
            mask = self.cur_mask.cuda()
        else:
            mask = self.cur_mask

        self.negative = Variable(output.data).expand(batch_size, batch_size, self.emb_size).gather(1, mask)
        self.positive = Variable(output.data)
        self.cur_mask = None

    def get_targets(self):
        assert self.positive is not None, 'Positive targets not set; needs a forward pass first'
        assert self.negative is not None, 'Negative targets not set; needs a forward pass first'
        return self.positive, self.negative

    def get_sens(self):
        # print(f"self.s_emb.shape: {self.s_emb.shape}")
        # print(f"self.s_emb.dim(): {self.s_emb.dim()}")
        # print(self.s_emb.mean(dim=2)[0].shape)
        if self.s_emb.dim() == 3:
            return self.s_emb
        else:
            return self.s_emb.mean(dim=2).mean(dim=0)

    def train(self, mode=True):
        super(SentimentAutoencoder,  self).train(mode)
        if self.encoder_hook is None:
            self.encoder_hook = self.seg_encoder.register_forward_hook(self.set_targets)
        return self

    def eval(self):
        super(SentimentAutoencoder, self).eval()
        if self.encoder_hook is not None:
            self.encoder_hook.remove()
            self.encoder_hook = None
        return self

def make_dir(dir):
    if os.path.exists(dir):
        print("dir exist!!")
        return
        shutil.rmtree(dir)
    os.makedirs(dir)
    print("make dir of {0}\n".format(dir)) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)

    # 實驗版本    
    parser.add_argument('--mver', help="version of model", type=str) # train out 的名字
    parser.add_argument('--sver', help="version of model", type=str) # seed 的名字
    parser.add_argument('--JASA_seed_num', help="", type=int) # , default=5
    parser.add_argument('--dataset', help="dataset source (YELP,AMAZON,FIQA_HEADLINE,FIQA_POST) ", type=str)
    parser.add_argument('--corpus_type', help='', type=str, default='sent')

    # 其他設定
    parser.add_argument('--round', help="iteration round", type=int, default=10)
    parser.add_argument('--epochs', help="Number of epochs (default: 10)", type=int, default=5) # 10 #adjust
    
    # 第一輪的情緒種子
    parser.add_argument('--sseed', help="version of model", type=str) # "g": general | "m": manmual | else | baseline
    # sentiment seed weight 預設的權重
    parser.add_argument('--sen_default_weight', help="origina: 0.5 | now: 0.1", type=str, default="0.1")

    # model
    parser.add_argument('-q', '--quiet', help="No information to stdout", action='store_true')
    parser.add_argument('--kmeans', help="Aspect embedding initialization with kmeans", action='store_true')
    parser.add_argument('--fix_w_emb', help="Fix word embeddings", action='store_true')
    parser.add_argument('--fix_a_emb', help="Fix aspect embeddings", action='store_true')
    parser.add_argument('--fix_s_emb', help="Fix sentiment embeddings", action='store_true')
    parser.add_argument('--attention', help="Use word attention", action='store_true')

    # model default
    parser.add_argument('--weight_type', help="type of sentiment seeds' weight(fix/trained)", type=str,default="fix")
    parser.add_argument('--sumout', help="Output file for summary generation (default: save)",type=str, default='save') # output.txt # adjust
    parser.add_argument('--min_len', help="Minimum number of non-stop-words in segment (default: 2)", type=int, default=2)
    parser.add_argument('--recon_method', help="Method of reconstruction (centr/init/fix/cos)",type=str, default='fix')
    parser.add_argument('--kmeans_iter', help="Number of times to re-run kmeans (default: 20)", type=int, default=20)
    parser.add_argument('--negative', help="Number of negative samples (default: 20)", type=int, default=20)
    parser.add_argument('--lr', help="Learning rate (default: 0.00001)", type=float, default=0.00001)
    parser.add_argument('--l', help="Orthogonality loss coefficient (default: 1)", type=float, default=1)
    parser.add_argument('--savemodel', help="File to save model in (default: don't)", type=str, default='')
    parser.add_argument('--seed', help="Random seed (default: system seed, -1)", type=int, default=-1)
    args = parser.parse_args()

    # torch.cuda.set_device(1) # adjust 1
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 

    if args.seed != -1:
        torch.manual_seed(args.seed)
        seed(args.seed)

    # input file & arguments

    # aspect setting
    asp_dict = pd.read_pickle('data/setting/aspect.pkl')    

    # number of aspect
    args_aspects = len(asp_dict[args.dataset])
    result_name = args.mver
    train_name = f'{args.dataset}_{args.corpus_type}'

    ## train & test
    f_pre = f"data/preprocessed/{args.dataset}"
    args_data      = f'{f_pre}/{train_name}' # train: Dataset name (without extension) (ex: YELP)
    args_test_data = f'{f_pre}/{train_name}_test.hdf5' # test: hdf5 file of test segments (ex: YELP_TEST)
    args_acc_file  = f'data/test/test_{train_name}.csv' # test label: "accurate file" (ex: test_YELP)    

    # make dir
    f_out = f"model_result/{result_name}"
    pdir = f'{f_out}/performance' # performance dir    
    sdir = f'{f_out}/seed' # seed dir   
    sumout_dir = f'{f_out}/result_ASSA' # sumout dir

    make_dir(pdir)
    make_dir(sdir)    
    make_dir(sumout_dir)    

    if not args.quiet:
        print('Loading data...')

    # word2vec
    id2word = {}
    word2id = {}
    fvoc = open(args_data + '_word_mapping.txt', 'r')
    for line in fvoc:
        word, id = line.split()
        id2word[int(id)] = word
        word2id[word] = int(id)
    fvoc.close()

    # train data
    f_train = f"{args_data}_train.hdf5"
    print(f"read {f_train}")
    f = h5py.File(f_train, 'r')
    batches = []
    original = []
    scodes = []
    for b in f['data']:
        batches.append(Variable(torch.from_numpy(f['data/' +  b][()]).long()))
        original.append(list(f['original/' + b][()]))
        scodes.append(list(f['scodes/' + b][()]))

    w_emb_array = f['w2v'][()]
    w_emb = torch.from_numpy(w_emb_array)
    vocab_size, emb_size = w_emb.size()
    f.close()

    test_batches = []
    test_labels = []
    test_labels_sen = []
    test_original = []
    test_scodes = []
    if args_test_data != '':
        f = h5py.File(args_test_data, 'r')
        for b in f['data']:
            test_batches.append(Variable(torch.from_numpy(f['data/' +  b][()]).long()))
            test_labels.append(Variable(torch.from_numpy(f['labels/' +  b][()]).long()))
            test_labels_sen.append(Variable(torch.from_numpy(f['labels_sen/' +  b][()]).long()))
            test_original.append(list(f['original/' + b][()]))
            test_scodes.append(list(f['scodes/' + b][()]))
    f.close()

    print("Loading acc...")
    acc_df = pd.read_csv(args_acc_file)
    id_asp_pair = dict()
    id_sen_pair = dict()
    for index, row in acc_df.iterrows():
        idx = str(row['scode'])
        text = row['sentence']
        label_asp = str(row['aspect'])
        label_sen = str(row['sentiment'])
        id_asp_pair[idx] = label_asp
        id_sen_pair[idx] = label_sen

    # begin of round
    update_round = 0
    while update_round < args.round:
        print("=" * 10 + f"round {update_round} begin" + "=" * 10)

        # aspect seeds
        args_aspect_seeds = f'seed/asp/asp_seeds_{args.dataset}.txt' # aspect: file that contains aspect seed words (overrides number of aspects)        
        print(f'read {args_aspect_seeds}')

        # 測seed品質用，不管哪一輪都是同一個
        if args.sseed == 'baseline': # baseline
            args_sentiment_seeds = f'{f_pre}/seed_baseline_{train_name}.txt'             
        else: # GCN
            args_sentiment_seeds = f'seed/sen/{args.dataset}/{args.sver}/GCN_seed_R1.txt'             
        print(f"read {args_sentiment_seeds}")

        if args.kmeans:
            # kmeans initialization (ABAE)
            if not args.quiet:
                print('Running k-means...')
            a_emb, _ = kmeans(w_emb_array, args_aspects, iter=args.kmeans_iter)
            a_emb = torch.from_numpy(a_emb)
            seed_w = None
            args.num_seeds = None
            ## SEN
            s_emb, _ = kmeans(w_emb_array, args.sentiments, iter=args.kmeans_iter)
            s_emb = torch.from_numpy(s_emb)
            seed_w_sen = None
            args.num_seeds_sen = None

        elif args_aspect_seeds != '':
            # seed initialization (MATE)
            fseed = open(args_aspect_seeds, 'r')
            fseed_sen = open(args_sentiment_seeds, 'r')
            aspects_ids = []
            sens_ids = []
            if args.recon_method == 'fix' or args.recon_method == 'init':
                seed_weights = []
                seed_weights_sen = []
            else:
                seed_weights = None
                seed_weights_sen = None

            for line in fseed:
                if args.recon_method == 'fix' or args.recon_method == 'init':
                    seeds = []
                    weights = []
                    for tok in line.split():
                        word, weight = tok.split(':')
                        if word in word2id:
                            seeds.append(word2id[word])
                            weights.append(float(weight))
                        else:
                            print(word,"aspect 找不到字ＱＱ")
                            seeds.append(0)
                            weights.append(0.0)
                    aspects_ids.append(seeds)
                    seed_weights.append(weights)
                else:
                    seeds = [word2id[word] if word in word2id else 0 for word in line.split()]
                    aspects_ids.append(seeds)
            fseed.close()

            ## SEN
            for line in fseed_sen:
                if args.recon_method == 'fix' or args.recon_method == 'init':
                    seeds_asp_sen = []
                    weights_asp_sen = []
                    # for each aspect
                    for asp_line in line.split(' | '):
                        seeds_sen = []
                        weights_sen = []
                        for tok in asp_line.split():
                            word, weight = tok.split(':')
                            if word in word2id:
                                if args.sen_default_weight:
                                    weight = args.sen_default_weight
                                seeds_sen.append(word2id[word])
                                weights_sen.append(float(weight))
                            else:
                                print(word,"sentiment 找不到字ＱＱ")
                                seeds_sen.append(0)
                                weights_sen.append(0.0)
                        # 只取前面幾個字
                        N = int(args.JASA_seed_num)
                        seeds_sen = seeds_sen[:N]
                        weights_sen = weights_sen[:N]
                        seeds_asp_sen.append(seeds_sen)
                        weights_asp_sen.append(weights_sen)

                    sens_ids.append(seeds_asp_sen)
                    seed_weights_sen.append(weights_asp_sen)
                else:
                    seeds_sen = [word2id[word] if word in word2id else 0 for word in line.split()]
                    sens_ids.append(seeds_sen)
            fseed_sen.close()  

            if seed_weights is not None:
                seed_w = torch.Tensor(seed_weights)
                seed_w /= seed_w.norm(p=1, dim=1, keepdim=True)
            else:
                seed_w = None
            
            ## SEN
            if seed_weights_sen is not None:
                seed_w_sen = torch.Tensor(seed_weights_sen)
                seed_w_sen /= seed_w_sen.norm(p=1, dim=2, keepdim=True)
            else:
                seed_w_sen = None

            if args.recon_method == 'centr':
                centroids = []
                for seeds in aspects_ids:
                    centroids.append(w_emb_array[seeds].mean(0))
                a_emb = torch.from_numpy(np.array(centroids))
                args_aspects = len(centroids)
                args.num_seeds = len(aspects_ids[0])
                ## SEN
                centroids_sen = []
                for seeds in sens_ids:
                    centroids_sen.append(w_emb_array[seeds].mean(0))
                s_emb = torch.from_numpy(np.array(centroids_sen))
                args.sentiments = len(centroids_sen)
                args.num_seeds_sen = len(sens_ids[0])
            else:
                clouds = []
                for seeds in aspects_ids:
                    clouds.append(w_emb_array[seeds])
                a_emb = torch.from_numpy(np.array(clouds))
                args_aspects = len(clouds)
                args.num_seeds = a_emb.size()[1]
                ## SEN
                clouds_asp_sen = []
                for asp_sens_ids in sens_ids:
                    clouds_sen = []
                    for seeds in asp_sens_ids:
                        # print(seeds)
                        clouds_sen.append(w_emb_array[seeds])
                    clouds_asp_sen.append(clouds_sen)
                # print(len(clouds_sen))
                s_emb = torch.from_numpy(np.array(clouds_asp_sen))
                # print(s_emb)
                # print(s_emb.shape)
                args.sentiments = len(clouds_sen)
                args.num_seeds_sen = s_emb.size()[2]
        else:
            a_emb = None
            s_emb = None
            seed_w = None
            seed_w_sen = None
            args.num_seeds = None
            args.num_seeds_sen = None


        if not args.quiet:
            print('Building model..')

        net = AspectAutoencoder(vocab_size, emb_size,
                num_aspects=args_aspects, neg_samples=args.negative,
                w_emb=w_emb, a_emb=a_emb, recon_method=args.recon_method, seed_w=seed_w,
                num_seeds=args.num_seeds, attention=args.attention, fix_w_emb=args.fix_w_emb,
                fix_a_emb=args.fix_a_emb)
        
        net_sen = SentimentAutoencoder(vocab_size, emb_size,
                num_aspects=args_aspects, num_sens=args.sentiments, neg_samples=args.negative,
                w_emb=w_emb, s_emb=s_emb, recon_method=args.recon_method, seed_w=seed_w_sen,
                num_seeds=args.num_seeds_sen, attention=args.attention, fix_w_emb=args.fix_w_emb,
                fix_s_emb=args.fix_s_emb)

        if torch.cuda.is_available():
            net = net.cuda()
            net_sen = net_sen.cuda()


        rec_loss = TripletMarginCosineLoss()
        if not args.fix_a_emb: # orthogonality loss is only used when training aspect matrix
            orth_loss = OrthogonalityLoss()

        ## ASP
        params = filter(lambda p: p.requires_grad, net.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr)
        ## SEN
        params_sen = filter(lambda p: p.requires_grad, net_sen.parameters())
        optimizer_sen = torch.optim.Adam(params_sen, lr=args.lr)

        if not args.quiet:
            print('Starting training...')
            print()

        start_all = time()

        performance_list = []
        for epoch in range(args.epochs):

            start = time()
            perm = permutation(len(batches))

            for i in range(len(batches)):
                inputs = batches[perm[i]]

                if inputs.shape[1] < args.min_len:
                    continue

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                ## ASP
                out, a_probs = net(inputs, perm[i])
                ## SEN
                out_sen, a_probs_sen = net_sen(inputs, a_probs, perm[i])

                positives, negatives = net.get_targets()
                positives_sen, negatives_sen = net_sen.get_targets()
                loss = rec_loss(out, positives, negatives) + rec_loss(out_sen, positives_sen, negatives_sen)

                ## ASP
                if not args.fix_a_emb:
                    aspects = net.get_aspects()
                    loss += args.l * orth_loss(aspects)
                ## SEN
                if not args.fix_s_emb:
                    sentiments = net_sen.get_sens()
                    loss += args.l * orth_loss(sentiments)

                optimizer.zero_grad()
                optimizer_sen.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_sen.step()

            if args.sumout != '':
                train_sumout = ''
                sumout = ''

            net.eval()
            net_sen.eval()

            ### added start
            for i in range(len(batches)):
                inputs = batches[i]
                ori = original[i]
                sco = scodes[i]
                if inputs.shape[1] < args.min_len:
                    continue
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                _, a_probs = net(inputs, i)
                _, a_probs_sen = net_sen(inputs, a_probs, i)

                for j in range(a_probs.size()[0]):
                    if args.sumout != '':
                        train_sumout += str(sco[j])                        
                        for a in range(a_probs.size()[1]):
                            train_sumout += '\t{0:.6f}'.format(a_probs.data[j][a])
                        for a in range(a_probs_sen.size()[1]):
                            train_sumout += '\t{0:.6f}'.format(a_probs_sen.data[j][a])
                        train_sumout += '\t' + str(ori[j]) + '\n'

            # 最後一個 epoch 時，匯出訓練資料的分類
            if epoch+1 == args.epochs:
                trianoutfname = '{0}/R{1}.train_out'.format(sumout_dir, update_round)
                fsum = open(trianoutfname, 'w')
                fsum.write(train_sumout)
                fsum.close()

            ### added end
            ## result of test data of epoch
            for i in range(len(test_batches)):
                inputs = test_batches[i]
                labels = test_labels[i]
                labels_sen = test_labels_sen[i]
                orig = test_original[i]
                sc = test_scodes[i]
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    labels_sen = labels_sen.cuda()

                _, a_probs = net(inputs, i)
                _, a_probs_sen = net_sen(inputs, a_probs, i)
                # print(a_probs_sen)

                for j in range(a_probs.size()[0]):
                    if args.sumout != '':
                        sumout += str(sc[j])
                        for a in range(a_probs.size()[1]):
                            sumout += '\t{0:.6f}'.format(a_probs.data[j][a])
                        for a in range(a_probs_sen.size()[1]):
                            sumout += '\t{0:.6f}'.format(a_probs_sen.data[j][a])
                        sumout += '\t' + str(orig[j]) + '\n'

            ## save test result
            if args.sumout != '':
                outfname = '{0}/E{1}_R{2}.sum'.format(sumout_dir, epoch+1,update_round)
                fsum = open(outfname, 'w') # open('{0}_{1}.sum'.format(args.sumout, epoch+1), 'w')
                fsum.write(sumout)
                fsum.close()

            net.train()
            net_sen.train()

            if not args.quiet:
                # outfname = '{0}_{1}.sum'.format(args.sumout, epoch+1)
                # print('Epoch {0}: {1}({2:6.2f}sec)'.format(epoch+1, count_acc(outfname, args_aspects, args.sentiments) , time() - start))
                print('\nEpoch {0}: ({1:6.2f}sec)'.format(epoch+1, time() - start)) 
                aacc, apre, arec, af1, sacc, spre, srec, sf1 = count_acc(outfname, args_aspects, args.sentiments) 
                performance_list.append({'Epoch':epoch+1, 'Time':time() - start, 'Loss': loss.item(), 
                    'AAccuracy':aacc, 'APrecision':apre, 'ARecall':arec, 'AF1_score':af1, 
                    'SAccuracy':sacc, 'SPrecision':spre, 'SRecall':srec, 'SF1_score':sf1})

        # save performance of test data
        with open(f'{pdir}/ASSA_R{update_round}.csv', 'w', newline='') as csvfile:
            fieldnames = ['Epoch', 'Time', 'Loss',
                'AAccuracy', 'APrecision', 'ARecall','AF1_score',
                'SAccuracy', 'SPrecision', 'SRecall','SF1_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for d in performance_list:
                writer.writerow(d)

        if not args.quiet:
            print('Finished training... ({0:.2f}sec)'.format(time() - start_all))
            print()

        if args.savemodel != '':
            if not args.quiet:
                print('Saving model...')
            torch.save(net.state_dict(), args.savemodel)    

        print("=" * 10 + f"round {update_round} finished" + "=" * 10)
        update_round = update_round + 1
