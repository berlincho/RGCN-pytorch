# Relational Graph Convolutional Networks (RGCN) Pytorch implementation
Pytorch-based implementation of RGCN for semi-supervised node classification on (directed) relational graphs. The code is adpated from [Kipf's Keras-based implementation](https://github.com/tkipf/relational-gcn). See details in [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (2017) [1].

The code for the *link prediction* task in [1] can be found in the following repository: https://github.com/MichSchli/RelationPrediction

### Dependencies
- Compatible with PyTorch 1.4.0 and Python 3.7.3.
- Dependencies can be installed using `requirements.txt`.

### Environment
- The implementation is supposed to train in the GPU enviornment. 
- I test all of the datasets with RGCN on GeForce RTX 2080 Ti and CPU with 128GB RAM.

### Dataset:
- RGCN use AIFB, MUTAG, and BGS as benchmark datasets for semi-supervised node classification.
- AIFB, MUTAG, and BGS are included in `data` directory. The datasets are adapted from RDF2Vec (2016).

### Training model (node classification):
We include early-stopping  mechanisms in `pytorchtools.py` to pick the optimal epoch.
- Install all the requirements from `requirements.txt.`
- AIFB: 
```shell
python run.py --data aifb --epochs 50 --bases 0 --hidden 16 --lr 0.01 --l2 0
```

- MUTAG: 
```shell
python run.py --data mutag --epochs 50 --bases 30 --hidden 16 --lr 0.01 --l2 5e-4
```

- BGS: 
```shell
python run.py --data bgs --epochs 50 --bases 40 --hidden 16 --lr 0.01 --l2 5e-4 --no_cuda
```
- AM:
```
python run.py --data am --epochs 50 --bases 40 --hidden 10 --lr 0.01 --l2 5e-4 --no_cuda
```
Note: Results depend on random seed and will vary between re-runs.
* `--bases` for RGCN basis decomposition
* `--data` denotes training datasets
* `--hidden` is the dimension of hidden GCN Layers
* `--lr` denotes learning rate
* `--l2` is the weight decay parameter of L2 regularization
* `--drop` is the dropout value for training GCN Layers
* Rest of the arguments can be listed using `python run.py -h`
