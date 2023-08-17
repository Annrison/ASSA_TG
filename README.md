### Description
This is the project for theis Weakly supervised Aspect-Based Sentiment Analysis with Tensor Graph Convolutional Network.

We propose a new framework called ASSA-TG, which improves the generation process of aspect-specific sentiment seeds. The original method only considers sequential relations between words. In our approach, TensorGCN is used to extract dependency relation and semantic similarity information to improve the quality of generated keywords.

### Related Links
+ About this paper
    + [Paper](https://drive.google.com/file/d/1ivmm44klPiairI98vRi43Xo8pWtFS1_o/view?usp=sharing)
    + [Slides](https://drive.google.com/file/d/1w2iHNBw1VCUHJYqNAWpGa4pGYRKcB5OO/view?usp=sharing)
+ References paper
    + [An Integration of TextGCN and Autoencoder into Aspect-based Sentiment Analysis (Tsai,2022)](https://link.springer.com/chapter/10.1007/978-3-031-12670-3_1)    
    + [Tensor graph convolutional networks for text classification (Liu, 2020) ](https://arxiv.org/abs/2001.05313)
    + [Graph Convolutional Networks for Text Classification (Yao, 2019)](https://arxiv.org/abs/1809.05679)
    + [Multiple Instance Learning Networks for Fine-Grained Sentiment Analysis (Angelidis, 2016)](https://link.springer.com/chapter/10.1007/978-3-031-12670-3_1)

+ References repo
    + [Tensor GCN model](https://github.com/THUMLP/TensorGCN)
    + [Text GCN model](https://github.com/yao8839836/text_gcn)
    + [MATE model](https://github.com/stangelid/oposum)

### How to Use

#### 0. Setup the Environment
`pip install requirements.txt`

#### 1. Download the Data

For training data, we use the restaurant reviews from Yelp and laptop reviews from Amazon, for testing data, we use the SemEval-2016 dataset, the data can be downloaded by the links below:

+ Training Data
    + [YELP](https://www.yelp.com/dataset/download)
    + [AMAZON](https://jmcauley.ucsd.edu/data/amazon/)
    + place in the folder `data/train/{dataset}`
+ Testing
    + [SemEval-2016](https://alt.qcri.org/semeval2016/task5/)
    + place in the folder `data/test`

#### 2. Preprocess Text Data

Use the `data/train/preprocess_data.ipynb` to preprocess the training & test data (lowercase, transfer url string and etc), and create word2vec model for ASSA model. Generated files includes:

1. `{dataset}_sent_5w.csv`: preprocessed training text
2. `{dataset}_sent_5w_wv.model`: word2vec model
3. `test/test_{dataset}_sent.csv`: preprocessed testing text

#### 3. Transfer Data format for ASSA model

The command would transfer data to format for ASSA model training, the result would be saved in the folder `data/preprocessed/{dataset}`, included:

1. `{dataset}_sent_test.hdf5`: test data in `.hdf5` format
2. `{dataset}_sent_train.hdf5`: train data in `.hdf5` format
3. `{dataset}_sent_word_counts.txt`: word frequency of train data
4. `{dataset}_sent_word_mapping.txt`: word & word ID of train data

```bash
cd data
#Training Data
python prep_hdf5_train.py --dataset="{dataset}" 
# Testing Data
python prep_hdf5_test.py --dataset="{dataset}"
# dataset variable can be `YELP` or `AMAZON`
```

#### 4. Train the ASSA model

Train the ASSA model, and the result would be saved in `model_result`, included the performance of each iteration and predicted results of text reviews. For the first round of the training, we use the general sentiment seeds in folder `{dataset}_sent_baseline`.

```bash
cd ..
python ./model/MATE.py \
--mver="{model_version}"
--sver="{seed_version}"
--JASA_seed_num= 10
--dataset= "{dataset}"
--round= 1
--epochs= 5
--sseed= "baseline"
```

Description of variables:
+ `mver`: model version
+ `sver`: sentiment seed version
+ `JASA_seed_num`: number of seeds
+ `dataset`: name of dataset (`YELP`/`AMAZON`)
+ `round`: number of iterate time
+ `epochs`: number of epoch of each iteration
+ `sseed`: whether to use baseline sentiment seed or not (`baseline`/`other`)

#### 5. Create Aspect-specific Sentiment Seeds by GCN

Building GCN graph and generate aspect-specific sentiment seeds of TensorGCN and TextGCN, the generated seeds would be in the folder `model_result/{dataset_model_version}`.

+ TensorGCN
Generated graph structure would be in the folder `TGCN/data_tgcn`

```bash
python TGCN.py \
--mver="{model_version}"
--dataset="{dataset}"
--round= 1
--graph="original"
--train_seed_num = 5
--thres= 0.3
--seed_type= `GCNonly`
```

+ TextGCN
Generated graph structure would be in the folder `TextGCN/data_textgcn`

```bash
python textGCN.py \
--mver="{model_version}"
--dataset="{dataset}"
--round= 1
--graph="original"
--train_seed_num = 5
--seed_type=`GCNonly`
```

Description of variables:
+ `mver`: model version
+ `dataset`: name of dataset (`YELP`/`AMAZON`)
+ `round`: iterate round of the seed generation
+ `graph`: the type of the edges constructing graph (`original`,`DP` or `DP+`)
+ `train_seed_num`: number of the training general sentiment seeds
+ `thres`: threshold of the word similarity of sematic graph
+ `seed_type`: whether to add general sentiment seeds to the final generated sentiment seeds (`GCNonly`/`add`)

+ Type of the edge in the graph
    In the original graph, we only add edges to the term pairs which have positive PMI, but for more experiment, we also try to exclude more edge to test if simplier graph can generate better seeds. Based on the papaer of Qiu,2016, we choose word pairs with specific dependency type. However, the result shows the seeds of original graph can improve the ASSA model the most. The edges of word pairs should be fulfill the the construction below:

    + `original`: Edges of word pairs with positive PMI
    + `DP`: Word pairs that have the following dependence types: `amod, case, nsubj, csubj, dobj, iobj, conj` and positive PMI
    + `DP+`:Word pairs that have the following dependence types: `amod, case, nsubj, csubj, dobj, iobj, conj, advmod, dep, cop, mark, nsubjpass, nmod, xcomp, xcomp, csubjpass, poss` and positive PMI

#### 6. Re-train the ASSA model by the specific-sentiment seeds

1. Move the generated seeds from the last step to the folder `seed/sen/{dataset}/{seed_version}`
2. Repeat the step 4 to train the ASSA model again, but change the `sver` variable to the name of new model version.