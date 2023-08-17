#!/usr/bin/env python

import sys
import h5py
import argparse
from time import time
import os

import pandas as pd
import numpy as np
from numpy.random import permutation, seed
from scipy.cluster.vq import kmeans
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_uniform
from torch.nn.utils import clip_grad_norm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from loss import TripletMarginCosineLoss, OrthogonalityLoss

def count_asp_acc(fname, asp_num):
    result_pair = []
    f = open(fname, 'r')
    for line in f:
        sp = line.split('\t')
        res_list = list(map(float, sp[1:asp_num+1]))
        result_pair.append({"scode": sp[0][2:-1], "asp":res_list.index(max(res_list))})
    f.close()
    y_test = []
    y_pred = []
    right_asp_cnt = 0
    for i in result_pair:
        y_test.append(str(i['asp']))
        y_pred.append(id_asp_pair[i['scode']])
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average = 'macro')
    rec = recall_score(y_test, y_pred, average = 'macro')
    f1 = f1_score(y_test, y_pred, average = 'macro')
    return acc, pre, rec, f1

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--domain', help="", type=str,default="FIQA_headline") 
    parser.add_argument('--corpus_type', help="", type=str,default="sent") 

    # parser.add_argument('--data', help="Dataset name (without extension)", type=str,default="./data/preprocessed/FIQA_HEADLINE_MATE") # YELP_MATE / FIQA_HEADLINE
    # parser.add_argument('--test_data', help="hdf5 file of test segments", type=str, default='./data/preprocessed/FIQA_headline_MATE_TEST.hdf5') # REST_MATE_TEST / FIQA_headline_MATE_TEST
    # parser.add_argument('--acc_file', help="accurate file", type=str, default='./data/test/test_FIQA_headline_sent.csv') # test_REST.txt
    # parser.add_argument('--aspect_seeds', help='file that contains aspect seed words (overrides number of aspects)',
    #         type=str, default='./seed/FIQA_headline_asp_seeds.txt') # REST_asp_seeds / FIQA_headline_asp_seeds

    parser.add_argument('--min_len', help="Minimum number of non-stop-words in segment (default: 2)", type=int, default=2)
    parser.add_argument('--aspects', help="Number of aspects (default: 10)", type=int, default=5) # original: 8
    parser.add_argument('--recon_method', help="Method of reconstruction (centr/init/fix/cos)",
            type=str, default='fix')
    parser.add_argument('--kmeans', help="Aspect embedding initialization with kmeans", action='store_true')
    parser.add_argument('--kmeans_iter', help="Number of times to re-run kmeans (default: 20)", type=int, default=20)
    parser.add_argument('--attention', help="Use word attention", action='store_true')
    parser.add_argument('--negative', help="Number of negative samples (default: 20)", type=int, default=20)
    parser.add_argument('--fix_w_emb', help="Fix word embeddings", action='store_true')
    parser.add_argument('--fix_a_emb', help="Fix aspect embeddings", action='store_true')
    parser.add_argument('--epochs', help="Number of epochs (default: 10)", type=int, default=100) # original: 10
    parser.add_argument('--lr', help="Learning rate (default: 0.00001)", type=float, default=0.00001)
    parser.add_argument('--l', help="Orthogonality loss coefficient (default: 1)", type=float, default=1)
    parser.add_argument('--savemodel', help="File to save model in (default: don't)", type=str, default='')
    parser.add_argument('--sumout', help="Output file for summary generation (default: output.txt)", type=str, default='./result_MATE/FIQA_headline') # original:output.txt
    parser.add_argument('--trainout_name', help="Train output file name (default: yelp)", type=str, default='result_MATE/FIQA_headline')
    parser.add_argument('-q', '--quiet', help="No information to stdout", action='store_true')
    parser.add_argument('--seed', help="Random seed (default: system seed, -1)", type=int, default=-1)
    parser.add_argument('--device', help="cuda visible devices", type=int, default=1)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.device)

    ## train & test
    args_data = f'./data/preprocessed/{args.domain.upper()}_MATE' 
    args_test_data = f'./data/preprocessed/{args.domain}_MATE_TEST.hdf5'
    args_acc_file = f'./data/test/test_{args.domain}_{args.corpus_type}.csv' 
    args_aspect_seeds = f'./seed/{args.domain}_asp_seeds.txt'

#     torch.cuda.set_device(1)

    if args.seed != -1:
        torch.manual_seed(args.seed)
        seed(args.seed)

    if not args.quiet:
        print('Loading data...')
    
    # word ID file (YELP_MATE_word_mapping.txt)
    id2word = {}
    word2id = {}
    fvoc = open(args_data + '_word_mapping.txt', 'r')
    for line in fvoc:
        word, id = line.split()
        id2word[int(id)] = word
        word2id[word] = int(id)
    fvoc.close()
    
    # training MATE data (YELP_MATE.hdf5)
    f = h5py.File(args_data + '.hdf5', 'r')
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
    test_original = []
    test_scodes = []
    if args_test_data != '':
        f = h5py.File(args_test_data, 'r')
        for b in f['data']:
            test_batches.append(Variable(torch.from_numpy(f['data/' +  b][()]).long()))
            test_labels.append(Variable(torch.from_numpy(f['labels/' +  b][()]).long()))
            test_original.append(list(f['original/' + b][()]))
            test_scodes.append(list(f['scodes/' + b][()]))
    f.close()
    
    # load labeled data (rest_test.txt)
    print("Loading acc...")

    id_asp_pair = dict()
    id_sen_pair = dict()

    acc_df = pd.read_csv(args_acc_file)
    for index, row in acc_df.iterrows():
        idx = str(row['scode'])
        text = row['sentence']
        label_asp = str(row['aspect'])
        id_asp_pair[idx] = label_asp

    # f = open(args.acc_file, 'r')
    # for line in f:
    #     idx, text, label_asp, label_sen = line.strip().split('\t')
    #     id_asp_pair[idx] = label_asp
    # f.close()



    if args.kmeans:
        # kmeans initialization (ABAE)
        if not args.quiet:
            print('Running k-means...')
        a_emb, _ = kmeans(w_emb_array, args.aspects, iter=args.kmeans_iter)
        a_emb = torch.from_numpy(a_emb)
        seed_w = None
        args.num_seeds = None
    elif args_aspect_seeds != '':
        # seed initialization (MATE)
        fseed = open(args_aspect_seeds, 'r')
        aspects_ids = []
        if args.recon_method == 'fix' \
                or args.recon_method == 'init':
            seed_weights = []
        else:
            seed_weights = None

        for line in fseed:
            if args.recon_method == 'fix' \
                    or args.recon_method == 'init':
                seeds = []
                weights = []
                for tok in line.split():
                    word, weight = tok.split(':')
                    if word in word2id:
                        seeds.append(word2id[word])
                        weights.append(float(weight))
                    else:
                        seeds.append(0)
                        weights.append(0.0)
                aspects_ids.append(seeds)
                seed_weights.append(weights)
            else:
                seeds = [word2id[word] if word in word2id else 0 for word in line.split()]
                aspects_ids.append(seeds)
        fseed.close()

        if seed_weights is not None:
            seed_w = torch.Tensor(seed_weights)
            seed_w /= seed_w.norm(p=1, dim=1, keepdim=True)
        else:
            seed_w = None

        if args.recon_method == 'centr':
            centroids = []
            for seeds in aspects_ids:
                centroids.append(w_emb_array[seeds].mean(0))
            a_emb = torch.from_numpy(np.array(centroids))
            args.aspects = len(centroids)
            args.num_seeds = len(aspects_ids[0])
        else:
            clouds = []
            for seeds in aspects_ids:
                clouds.append(w_emb_array[seeds])
            a_emb = torch.from_numpy(np.array(clouds))
            args.aspects = len(clouds)
            args.num_seeds = a_emb.size()[1]
    else:
        a_emb = None
        seed_w = None
        args.num_seeds = None


    if not args.quiet:
        print('Building model..')

    net = AspectAutoencoder(vocab_size, emb_size,
            num_aspects=args.aspects, neg_samples=args.negative,
            w_emb=w_emb, a_emb=a_emb, recon_method=args.recon_method, seed_w=seed_w,
            num_seeds=args.num_seeds, attention=args.attention, fix_w_emb=args.fix_w_emb,
            fix_a_emb=args.fix_a_emb)

    if torch.cuda.is_available():
        net = net.cuda()

    rec_loss = TripletMarginCosineLoss()
    if not args.fix_a_emb: # orthogonality loss is only used when training aspect matrix
        orth_loss = OrthogonalityLoss()

    params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

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

            out, a_probs = net(inputs, perm[i])

            positives, negatives = net.get_targets()
            loss = rec_loss(out, positives, negatives)

            if not args.fix_a_emb:
                aspects = net.get_aspects()
                loss += args.l * orth_loss(aspects)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if args.sumout != '':
            train_sumout = ''
            sumout = ''

        net.eval()

        # 產生 training 的 aspect output
        for i in range(len(batches)):
            inputs = batches[i]
            ori = original[i]
            sco = scodes[i]
            if inputs.shape[1] < args.min_len:
                continue
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            _, a_probs = net(inputs, i)
            
            for j in range(a_probs.size()[0]):
                if args.sumout != '':
                    train_sumout += str(sco[j])
                    for a in range(a_probs.size()[1]):
                        train_sumout += '\t{0:.6f}'.format(a_probs.data[j][a])
                    train_sumout += '\t' + str(ori[j]) + '\n'
        
        # 最後一個 epoch 時，匯出訓練資料的 aspect 分類
        if epoch+1 == args.epochs:
            fsum = open('{0}.train_out'.format(args.trainout_name), 'w')
            fsum.write(train_sumout)
            fsum.close()
            print("save aspect predict to {}".format(args.trainout_name))
            
        for i in range(len(test_batches)):
            inputs = test_batches[i]
            labels = test_labels[i]
            orig = test_original[i]
            sc = test_scodes[i]
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            _, a_probs = net(inputs, i)

            for j in range(a_probs.size()[0]):
                if args.sumout != '':
                    sumout += str(sc[j])
                    for a in range(a_probs.size()[1]):
                        sumout += '\t{0:.6f}'.format(a_probs.data[j][a])
                    sumout += '\t' + str(orig[j]) + '\n'
                    
        # save MATE's training result ((yelp.train_out))
        # 每個 epoch，匯出測試資料的 aspect 分類
        if args.sumout != '':
            fsum = open('{0}_{1}.sum'.format(args.sumout, epoch+1), 'w')
            fsum.write(sumout)
            fsum.close()

        net.train()

        if not args.quiet:
            outfname = '{0}_{1}.sum'.format(args.sumout, epoch+1)
            # print('Epoch {0}: -ASPECT- {1}({2:6.2f}sec)'.format(epoch+1, count_asp_acc(outfname, args.aspects) , time() - start))
            acc, pre, rec, f1 = count_asp_acc(outfname, args.aspects)
            print(f"Epoch {epoch+1}: -ASPECT- Accuracy: {acc:.5f} Precision: {pre:.5f} Recall: {rec:.5f} F1: {f1:.5f} Loss:{loss.item():.5f} ({(time() - start):.2f}sec)")
            performance_list.append({'Epoch':epoch+1, 'Time':time() - start, 'Loss': loss.item(), 
                                'AAccuracy':acc, 'APrecision':pre, 'ARecall':rec, 'AF1_score':f1})

    # 儲存預測結果
    with open(f'result_MATE/{args.domain}/performance.csv', 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Time', 'Loss', 'AAccuracy', 'APrecision', 'ARecall','AF1_score']
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