'''
3.2 extract student model
use to prove knowledge distillation process

exec: 
$ python extract_student_model.py

output folder: extraction_student

dependency:
$ pip install torch torchvision

version:
python: 3.9.21
torch==2.5.1
torchaudio==2.5.1
torchvision==0.20.1

device:
CPU: intel i7-12700
RAM: 64 GB
GPU: NVDIA GeForce RTX 3080 Ti
cuda: 12.4
cuDNN: 9.1.0
'''
import os
import argparse
import pickle
import torch
from torch import nn
from torchvision import datasets, transforms

from model import SoftDecisionTree, InnerNode, LeafNode
from model import TeacherModel

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',  # 設定 batch size 為 1
                    help='input batch size for training (default: 64)')
parser.add_argument('--input-dim', type=int, default=28*28, metavar='N',
                    help='input dimension size(default: 28 * 28)')
parser.add_argument('--output-dim', type=int, default=10, metavar='N',
                    help='output dimension size(default: 10)')
parser.add_argument('--max-depth', type=int, default=8, metavar='N',
                    help='maximum depth of tree(default: 8)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',  # 設定 epochs 為 1
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lmbda', type=float, default=0.1, metavar='LR',
                    help='temperature rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def add_operation_log_to_nodes(node):
    if isinstance(node, InnerNode) or isinstance(node, LeafNode):
        node.operation_log = []
        if isinstance(node, InnerNode):
            node.left_child_id = id(node.left) if node.left else None
            node.right_child_id = id(node.right) if node.right else None
            if not node.leaf:
                add_operation_log_to_nodes(node.left)
                add_operation_log_to_nodes(node.right)


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    try:
        os.makedirs('./data')
    except:
        print('directory ./data already exists')

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    # full_dataset = datasets.MNIST('./data', train=False, download=True,
    #                               transform=transforms.Compose([
    #                                   transforms.ToTensor(),
    #                                   transforms.Normalize((0.1307,), (0.3081,))
    #                               ]))
    # indices = [i for i, (img, target) in enumerate(full_dataset) if target == 9]
    # subset_dataset = torch.utils.data.Subset(full_dataset, indices)

    # test_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    softdecisiontree_updated = torch.load('student_updated.pkl')

    add_operation_log_to_nodes(softdecisiontree_updated.root)


    if args.cuda:
        softdecisiontree_updated.cuda()

    softdecisiontree_updated.test_one_graph(test_loader, 1)

    # softdecisiontree_updated.treiverse_and_extract_to_json(test_loader)