# SWIFT
Code repository for the SIGMOD 2026 under review paper:
"SWIFT: Enabling Large-Scale Temporal Graph Learning on a Single Machine"

```
/
├── config/
├── preprocess/
│   ├── data_process.py --for processing raw graph data (new)
│   ├── root_data_process.py --for processing raw graph data (new)
│   └── swift.py --for SWIFT Preprocessing (new)
├── sampler/
│   ├── build/
│   ├── sampler.py
│   ├── sampler_gpu.py --for GPU Sampler (new)
│   └── setup.py
├── swift-dgl/
├── README.md
├── down.sh
├── down_wiki.sh
├── feat_buffer.py --for Bucket Optimization (new)
├── gen_graph.py
├── layers.py
├── memorys.py
├── modules.py
├── pre_fetch.py --for Pipeline Optimization (new)
├── sample_output.txt
├── setup.py
├── train.py
└── utils.py
```
## Requirements
- python >= 3.8
- pytorch >= 1.12.1
- numpy >= 1.24.3
- dgl >= 0.9.1

Since SWIFT is based on the CUDA compilation process of the DGL framework, it is necessary to compile the given DGL version to obtain some CUDA functions used during SWIFT training.

```
cd swift/swift-dgl
bash build.sh
```


## Prepare Datasets
We use five datasets in the paper: LastFM, Wiki-Talk, Stack-Overflow, Bitcoin, GDELT.
Among them, LastFM and GDELT can be downloaded via swift/down.sh.
Wiki-Talk and Stack-Overflow need to be downloaded from http://snap.stanford.edu/data/wiki-talk-temporal.html and https://snap.stanford.edu/data/sx-stackoverflow.html respectively. Bitcoin can be downloaded from https://networkrepository.com/soc-bitcoin.php.
Please note that after downloading the Wiki-Talk, Stack-Overflow, and Bitcoin datasets, the raw data needs to be processed to obtain temporal graph pattern data:

```
python /root/swift/preprocess/root_data_process.py --data dataset_name --txt graph_data_source_file
python /root/swift/preprocess/data_process.py --data dataset_name
```

## Configuration
We provide some model configuration files in swift/config, with specific configuration descriptions as follows (content sourced from the open-source framework TGL):
```
sampling:
  - layer: <number of layers to sample>
    neighbor: <a list of integers indicating how many neighbors are sampled in each layer>
    strategy: <'recent' that samples most recent neighbors or 'uniform' that uniformly samples neighbors form the past>
    prop_time: <False or True that specifies wherether to use the timestamp of the root nodes when sampling for their multi-hop neighbors>
    history: <number of snapshots to sample on>
    duration: <length in time of each snapshot, 0 for infinite length (used in non-snapshot-based methods)
    num_thread: <number of threads of the sampler>
memory: 
  - type: <'node', we only support node memory now>
    dim_time: <an integer, the dimension of the time embedding>
    deliver_to: <'self' that delivers the mails only to involved nodes or 'neighbors' that deliver the mails to neighbors>
    mail_combine: <'last' that use the latest latest mail as the input to the memory updater>
    memory_update: <'gru' or 'rnn'>
    mailbox_size: <an integer, the size of the mailbox for each node>
    combine_node_feature: <False or True that specifies whether to combine node features (with the updated memory) as the input to the GNN.
    dim_out: <an integer, the dimension of the output node memory>
gnn:
  - arch: <'transformer_attention' or 'identity' (no GNN)>
    layer: <an integer, number of layers>
    att_head: <an integer, number of attention heads>
    dim_time: <an integer, the dimension of the time embedding>
    dim_out: <an integer, the dimension of the output dynamic node embedding>
train:
  - epoch: <an integer, number of epochs to train>
    batch_size: <an integer, the batch size (of edges); for multi-gpu training, this is the local batchsize>
    reorder: <(optional) an integer that is divisible by batch size the specifies how many chunks per batch used in the random chunk scheduling>
    lr: <floating point, learning rate>
    dropout: <floating point, dropout>
    att_dropout: <floating point, dropout for attention>
    all_on_gpu: <False or True that decides if the node/edge features and node memory are completely stored on GPU>
```
Corresponding to the paper, we provide initial configuration files for the TGAT, TGN, and TimeSGN models under 1-layer and 2-layer message passing in swift/config.

## Preprocessing
The preprocessing process mainly involves bucket construction operations. It is necessary to complete the dataset preparation and model configuration according to the Prepare Datasets step and the file configuration step beforehand. Execute:
```
python swift/preprocess/swift.py --data dataset_name --config model_config_file_path
```

Please note that for different models, preprocessing only needs to be performed once under the same sampling configuration. However, different sampling parameters (such as different layer numbers or different fanouts) require re-execution of the preprocessing operation.

## Train

After completing the preprocessing operation, you can run the training program by specifying the dataset name and model configuration file path:
```
python swift/train.py --data dataset_name --config model_config_file_path
```

