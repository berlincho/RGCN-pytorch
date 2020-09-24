import os
from models import *
from data_utils import *
from scipy.sparse import identity
import logging
import torch
import argparse
import warnings
# import EarlyStopping
from pytorchtools import EarlyStopping
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Enables CUDA training.')
parser.add_argument('--validation', action='store_true', default=False,
                    help='Run validation data.')
#parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--l2', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hop', type=int, default=2,
                    help='Number of hops.')
parser.add_argument('--drop', type=float, default=0,
                    help='Dropout of SAHO')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--bases', type=int, default=30,
                    help='R-GCN bases')
parser.add_argument('--data', type=str, default="mutag",
                    help='dataset.')
args = parser.parse_args()
print(args)
args.using_cuda = not args.no_cuda and torch.cuda.is_available()
if args.using_cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

def input_data(dirname='./data'):
    data = None
    if os.path.isfile(dirname + '/' + args.data + '_' + str(args.hop) + '.pickle'):
        with open(dirname + '/' + args.data + '_' + str(args.hop) + '.pickle', 'rb') as f:
            data = pkl.load(f)
    else:
        with open(dirname + '/' + args.data + '_' + str(args.hop) + '.pickle', 'wb') as f:
            # Data Loading...    
            A, X, y, labeled_nodes_idx, train_idx, test_idx, rel_dict, train_names, test_names = load_data(args.data, args.hop)
            data = {'A': A,
                    'y': y,
                    'train_idx': train_idx,
                    'test_idx': test_idx,
            }
            pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
    return data['A'], data['y'], data['train_idx'], data['test_idx']

# Load Data
A, y, train_idx, test_idx = input_data()
# Get dataset splits
y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y, train_idx, test_idx, args.validation)
num_nodes = A[0].shape[0]
num_rel = len(A)

# Adjacency matrix normalization
A = row_normalize(A)

# Create Model
model = RelationalGraphConvModel(input_size=num_nodes, hidden_size=args.hidden, output_size=y_train.shape[1], num_bases=args.bases, num_rel=num_rel, num_layer=2, dropout=args.drop, featureless=True, cuda=args.using_cuda)
print('Loaded %s dataset with %d entities, %d relations and %d classes' % (args.data, num_nodes, num_rel, y_train.shape[1]))
        
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2) 

# initialize the early_stopping object
early_stopping = EarlyStopping(patience=10, verbose=True)

if args.using_cuda:
    print("Using the GPU")
    model.cuda()

    
test_loss = []
test_acc = []
X = None
# Start training
for i in range(args.epochs):
    model.train()
    emb_train = model(A, X)
    scores = emb_train[idx_train]
    labels_train = torch.LongTensor(np.array(np.argmax(y_train[idx_train], axis=-1)).squeeze())
    labels_train = labels_train.cuda() if args.using_cuda else labels_train
    loss = criterion(scores, labels_train)
    print ("Epoch: {epoch}, Training Loss on {num} training data: {loss}".format(epoch=i, num=len(idx_train), loss=str(loss.item())))
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Do validation
    if args.validation:
        print("----------------------")
        model.eval()
        with torch.no_grad():
            correct = 0
            emb_val = model(A, X)
            scores = emb_val[idx_val]
            _, predicted = torch.max(scores.data, 1)
            labels_valid = torch.LongTensor(np.array(np.argmax(y_val[idx_val], axis=-1)).squeeze())
            labels_valid = labels_valid.cuda() if args.using_cuda else labels_valid
            correct += (predicted == labels_valid).sum().item()
            print('    Accuracy of the network on the {num} validation data: {acc} %'.format(num=len(idx_val), acc=100* correct / labels_valid.size(0)))

    # Do Testing
    if not args.validation:
        print("----------------------")
        model.eval()
        with torch.no_grad():
            correct = 0
            emb_test = model(A, X)
            scores = emb_test[idx_test]
            _, predicted = torch.max(scores.data, 1)
            labels_test = torch.LongTensor(np.array(np.argmax(y_test[idx_test], axis=-1)).squeeze())
            labels_test = labels_test.cuda() if args.using_cuda else labels_test
            correct += (predicted == labels_test).sum().item()
            loss = criterion(scores, labels_test)
            acc = 100* correct / labels_test.size(0)
            test_loss.append(loss.item())
            test_acc.append(acc)
            print('Accuracy of the network on the {num} test data: {acc} %, loss: {loss}'.format(num=len(idx_test), acc=acc, loss=loss.item()))   
            
            #if early_stopping, it will make a checkpoint of the current model
            early_stopping(loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
    