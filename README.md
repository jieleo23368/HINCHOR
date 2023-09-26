# HINCHOR
Counterfactual Learning for Higher-Order Relation Prediction in Heterogeneous Information Networks

## Requirements
-python==3.6.9  

-torch==1.10.0  

-torch_geometric==2.0.3  

-networkx==2.5.1  

-scipy==1.6.2  

-numpy==1.19.4   

-scikit_learn==0.24.2  

-scikit_network==0.20.0  


## Datasets
DBLP(https://github.com/cynricfu/MAGNN) is a computer science bibliography website. We extract a subset of DBLP which contains 14376 papers, 14475 authors, 20 conferences and 8920 terms. 

DouBan(https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding), we extract a subset of DouBan Book which contains 3000 books, 2141 authors, 615 publisher.

Aminer(https://www.aminer.cn/citation) is a academic social network. We extract a subset of Aminer which contains 5000 papers, 13143 authors and 280 conferences. 

IMDB(https://grouplens.org/datasets/movielens/100k/) is a movie review database. We construct heterogeneous network with 943 users, 1682 movies, 5 ratings and 19 genres.


## Usage
Following the command to run the codes(For all datasets the commands are same).
```
python3 main.py --seed 1
```






