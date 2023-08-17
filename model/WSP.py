import re
import argparse
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from collections import defaultdict
import csv

def line_to_words(line, min_len, max_len, stop_words=None):
    """
    Reads a line of text (sentence) and returns a list of tokenized EDUs
    """

    # clean sentence and break it into EDUs
    clean_line = clean_str(line.strip())

    word_list = word_tokenize(clean_line)

    if stop_words is not None:
        words = [word for word in word_list]# if word not in stop_words]

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
    stop_words = args.stop_words

    data = []
    labels_asp = []
    labels_sen = []
    scodes = []
    original = []

    doc_cnt = 0

    f = open(file, 'r')
    for line in f:
        idx, text, label_asp = line.strip().split('\t')

        words = line_to_words(text, 1, 10000, stop_words=stop_words)
        if words is None:
            continue
        # for word in words:
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

    f.close()

    print('Number of documents:', doc_cnt)
    print('Number of aspects:', num_asp)
    print('Vocabulary size:', len(word2id))

    return data, labels_asp, labels_sen, scodes, original

def clean_str(string):
    """
    String cleaning
    """
    string = string.lower()
    string = re.sub(r"\n", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"&#34;", " ", string)
    string = re.sub(r"(http://)?www\.[^ ]+", " _url_ ", string)
    # string = re.sub(r"[^a-z0-9$\'_]", " ", string)
    # string = re.sub(r"^[0-9].", "", string) # 新加的
    # string = re.sub(r"[^\u4E00-\u9FA50-9$\'_]", " ", string) # 新加的

    string = re.sub(r"_{2,}", "_", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\$+", " $ ", string)
    string = re.sub(r"rrb", " ", string)
    string = re.sub(r"lrb", " ", string)
    string = re.sub(r"rsb", " ", string)
    string = re.sub(r"lsb", " ", string)
    string = re.sub(r"(?<=[a-z])I", " I", string)
    string = re.sub(r"(?<= )[0-9]+(?= )", "NUM", string)
    string = re.sub(r"(?<= )[0-9]+$", "NUM", string)
    string = re.sub(r"^[0-9]+(?= )", "NUM", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def count_acc(fname, asp_num, sen_num):
    result_pair = []
    f = open(fname, 'r')
    for line in f:
        sp = line.split('\t')
        asp_res_list = list(map(float, sp[1:asp_num+1]))
        sen_res_list = list(map(float, sp[asp_num+1:asp_num+sen_num+1]))
        result_pair.append({"scode": sp[0][2:-1], 
                            "asp":asp_res_list.index(max(asp_res_list)), 
                            "sen":sen_res_list.index(max(sen_res_list))})
    f.close()
    y_test_asp = []
    y_pred_asp = []
    y_test_sen = []
    y_pred_sen = []
    for i in result_pair:
        y_test_asp.append(str(i['asp']))
        y_pred_asp.append(id_asp_pair[i['scode']])
        y_test_sen.append(str(i['sen']))
        y_pred_sen.append(id_sen_pair[i['scode']])
    a_acc, a_pre, a_rec, a_f1 = count_aprf(y_test_asp, y_pred_asp)
    s_acc, s_pre, s_rec, s_f1 = count_aprf(y_test_sen, y_pred_sen)
    return(f"-ASPECT- Accuracy: {a_acc:.5f} Precision: {a_pre:.5f} Recall: {a_rec:.5f} F1: {a_f1:.5f} -SENTIMENT- Accuracy: {s_acc:.5f} Precision: {s_pre:.5f} Recall: {s_rec:.5f} F1: {s_f1:.5f}")

def count_aprf(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average = 'macro')
    rec = recall_score(y_test, y_pred, average = 'macro')
    f1 = f1_score(y_test, y_pred, average = 'macro')
    return acc,pre,rec,f1


if __name__=="__main__":

    parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--graph', help="Graph Type ('yelp','DP' or 'DP+')", type=str,default="yelp")
    parser.add_argument('--ASSA_round', help="number of update round of sentiment seed words", type=int,default=0)

    parser.add_argument('--MATE_data', help='data in appropriate format', type=str, default='./result_MATE/rest_30.sum')
    parser.add_argument('--domain', help='category of domain', type=str, default="REST")
    # parser.add_argument('--sentiment_seeds', help='file that contains all sentiment seed words', type=str, default='./seed/REST_GCN_all.txt')
    parser.add_argument('--aspects', help='number of aspect', type=int, default=3)
    parser.add_argument('--sentiments', help='number of sentiment', type=int, default=2)
    parser.add_argument('--vocab', help='vocabulary file (from train set)', type=str, default='./data/preprocessed/YELP_MATE_word_mapping.txt')
    parser.add_argument('--batch_size', help='maximum number of segments per batch (default: 50)', type=int, default=50)
    parser.add_argument('--padding', help='padding around each sentence (default: 0)', type=int, default=0)
    parser.add_argument('--seed', help='random seed (default: 1)', type=int, default=1)
    args = parser.parse_args()

    print("count accuracy WSP Round :{0}".format(args.ASSA_round))
    print("args :graph {0}  ASSA_round{1}".format(args.graph, args.ASSA_round))

    print("Loading acc...")
    f = open(f"./data/test/test_rest.txt", 'r')

    id_text_pair = dict()
    id_asp_pair = dict() # real aspect
    id_sen_pair = dict() # real sentiment
    for line in f:
        idx, text, label_asp, label_sen = line.strip().split('\t')
        id_text_pair[idx] = text
        id_asp_pair[idx] = label_asp
        id_sen_pair[idx] = label_sen
    f.close()

    asp_dict = {
        "REST": ["AMB", "DRI", "FOD", "LOC", "SRV"],
        "LAPTOP": ["SUPPORT", "OS", "DISPLAY", "BATTERY", "COMPANY", "MOUSE", "SOFTWARE", "KEYBOARD"]
    }
    asp_name_list = asp_dict[args.domain.upper()]

    args.stop_words = set(stopwords.words('english'))

    word2id = {}
    fvoc = open(args.vocab, 'r')
    for line in fvoc:
        word, id = line.split()
        word2id[word] = int(id)
    fvoc.close()
    word2id['not'] = 1000000

    # load seed words (all)
    performance_list = []
    sentiment_seedsf = "./seed/{0}/{1}/GCN_all_R{2}.txt".format(args.domain, args.graph, args.ASSA_round)

    for percent in range(0, 50, 1):
        fseed_sen = open(sentiment_seedsf, 'r')
        all_seed = {} # {'AMB':{20:0.5, 197:0.5 ...}...}
        count = 0 # tocheck
        for line in fseed_sen:
            all_seed[asp_name_list[count]]={}
            for asp_line in line.split(' | '): # pos, neg
                for tok in asp_line.split(): # word:polarity
                    word, weight = tok.split(':')
                    if word in word2id:
                        if abs(float(weight))*100 < percent: # skip if abs(polarity) < percent
                            continue
                        all_seed[asp_name_list[count]][word2id[word]] = float(weight)
            count += 1
        fseed_sen.close()
        # print(len(all_seed[asp_name_list[0]]))

        # MATE predict result (test data)
        f = open(args.MATE_data, "r")
        result_pair = []
        for line in f:
            sp = line.split('\t')
            res_list = list(map(float, sp[1:len(asp_name_list)+1]))
            result_pair.append({"scode": int(sp[0][2:-1]), 
                                "aspect": res_list.index(max(res_list)), 
                                "text":sp[len(asp_name_list)+1].strip()})
        f.close()

        all_doc = defaultdict() # {"DRI":[doc1,doc2...]...}
        for i in result_pair:
            all_doc.setdefault(asp_name_list[i['aspect']], []).append(i['text'][2:-1]) # old [2:-3] 

        # for i in asp_name_list:
        #     print(f"{i}: {len(all_doc[i])}")
        
        # 計算每個doc的polarity值
        sentiment_result_dict = dict() # {"scode":polarity}
        for text in result_pair:
            words = line_to_words(text['text'][2:-1], 1, 10000, stop_words=args.stop_words)
            if words is None:
                continue

            # 將上面的words轉換為 word id
            seg_ids = [word2id[word] if word in word2id else 1 for word in words] # id=1 if not in word2id 
            sentiment_value = 0 # sentiment value of doc
            for s_index, s in enumerate(seg_ids): # s_index: word position, s: word id
                if s in all_seed[asp_name_list[text["aspect"]]]:
                    # not 情緒反轉
                    if 1000000 in seg_ids[max(0, s_index-2):s_index]: # 1000000 是 not 的 id (前後各一個字的情緒都會反轉)
                        sentiment_value += all_seed[asp_name_list[text["aspect"]]][s]*-1 # polarity * -1 
                    else:
                        sentiment_value += all_seed[asp_name_list[text["aspect"]]][s]
            sentiment_result_dict[text['scode']] = sentiment_value

        # 依據計算出的polarity，計算每個doc是正面還是負面
        y_test_sen = [] # real label
        y_pred_sen = [] # predict result
        for idx, value in sentiment_result_dict.items(): # idx: scode, value: polarity
            y_test_sen.append(id_sen_pair[str(idx)]) # id_sen_pair: real label
            if value >= 0:
                y_pred = "0"
            else:
                y_pred = "1"
            y_pred_sen.append(y_pred)
            # if y_pred != id_sen_pair[str(idx)]:
            #     print(f"Text: {id_text_pair[str(idx)]} , ACC: {id_sen_pair[str(idx)]} , PRED: {str(value)}")
        s_acc, s_pre, s_rec, s_f1 = count_aprf(y_test_sen, y_pred_sen) # real label, predict label
        print(f"Threshold:{percent}, len(word):{len(all_seed[asp_name_list[0]])},  -SENTIMENT- Accuracy: {s_acc:.5f} Precision: {s_pre:.5f} Recall: {s_rec:.5f} F1: {s_f1:.5f}")
        performance_list.append({'Threshold':percent/100+0.5, 'Accuracy':s_acc, 'Precision':s_pre, 'Recall':s_rec, 'F1_score':s_f1})

    with open(f'performance/{args.domain}/{args.graph}/WSP_R{args.ASSA_round}.csv', 'w', newline='') as csvfile:
        fieldnames = ['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for d in performance_list:
            writer.writerow(d)