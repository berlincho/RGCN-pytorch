import argparse
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="testrun",
                    help='Provide a test name.')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Enables CUDA training.')
parser.add_argument('--validation', action='store_true', default=False,
                    help='Run validation data.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--l2', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hop', type=int, default=1,
                    help='Number of hops.')
parser.add_argument('--drop', type=float, default=0,
                    help='Dropout of RGCN')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--bases', type=int, default=30,
                    help='R-GCN bases')
parser.add_argument('--data', type=str, default="mutag",
                    help='dataset.')

args = parser.parse_args()
args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
args.using_cuda = not args.no_cuda and torch.cuda.is_available()
if args.using_cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True