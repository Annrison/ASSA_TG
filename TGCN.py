import argparse
from subprocess import call
import pandas as pd
import os
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--mver', help="version of model", type=str) # seed_train10
    parser.add_argument('--train_seed_num', help='number of sentiment seed', type=str, default="10")
    parser.add_argument('--thres', help='', type=str, default='0.3') 

    parser.add_argument('--round', help="iteration round", type=int, default=1)
    parser.add_argument('--graph', help="Graph Type ('original','DP' or 'DP+')", type=str, default="original") # graph=> original
    parser.add_argument('--dataset', help="dataset source (YELP,AMAZON,FIQA) ", type=str, default="FIQA_HEADLINE")
    parser.add_argument('--corpus_type', help='doc or sent', type=str, default="sent")    
    parser.add_argument('--seed_type', help='type of sentiment seed', type=str, default="GCNonly")

    
    # model
    # parser.add_argument('--epochs', help="Number of epochs (default: 10)", type=int, default=10) # 10 #adjust
    parser.add_argument('--weight_type', help="type of sentiment seeds' weight(fix/trained)", type=str,default="fix")
    args = parser.parse_args()

    asp_dict = pd.read_pickle('./data/setting/aspect.pkl') 

    print("args.dataset",args.dataset)
    # input file & arguments
    args_aspects = len(asp_dict[args.dataset])

    # make dir of round # 0608 (之後可以拿掉，新增放seed的資料夾)
    sdir = './model_result/{0}'.format(args.mver) 
    if os.path.exists(sdir):
        shutil.rmtree(sdir)
    os.makedirs(sdir)
    print("make dir of {0}".format(sdir))

    update_round = 0
    while update_round < args.round:
        print("=" * 10 + f"round {update_round} begin" + "=" * 10)
        
        # # TGCN
        print("TGCN graph...")
        call(["python", "./TGCN/build_graph_tgcn.py", 
                "--mver", args.mver, 
                "--dataset", args.dataset,
                "--graph", args.graph,                 
                "--ASSA_round", str(update_round),
                "--train_seed_num", args.train_seed_num,
                "--corpus_type", args.corpus_type,
                "--thres", args.thres,
                ])
    
        print("generate sentiment seed...")
        call(["python", "./TGCN/train_tgcn.py", 
            args.dataset, 
            args.graph, 
            str(update_round), 
            args.mver, 
            args.weight_type,
            args.seed_type,
            args.train_seed_num
           ])

        print("=" * 10 + f"round {update_round} finished" + "=" * 10)
        update_round = update_round + 1

