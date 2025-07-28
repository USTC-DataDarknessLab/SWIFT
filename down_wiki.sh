#!/bin/bash

wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edge_features.pt
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edges.csv  
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/ext_full.npz
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/int_full.npz
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/int_train.npz
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/labels.csv