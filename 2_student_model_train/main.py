from __future__ import print_function
import os
import argparse
import pickle
import torch
from torch import nn
from torchvision import datasets, transforms

from model import SoftDecisionTree

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--input-dim', type=int, default=28*28, metavar='N',
                    help='input dimension size(default: 28 * 28)')
parser.add_argument('--output-dim', type=int, default=10, metavar='N',
                    help='output dimension size(default: 10)')
parser.add_argument('--max-depth', type=int, default=8, metavar='N',
                    help='maximum depth of tree(default: 8)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 40)')
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


class TeacherModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(TeacherModel, self).__init__()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return x
    


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

    teachermodel = torch.load('teacher.pkl')

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

           

    model = SoftDecisionTree(args)

    if args.cuda:
        model.cuda()

    for epoch in range(1, args.epochs + 1):
        model.train_kd_(train_loader, epoch, teachermodel)
        # model.train_(train_loader, epoch)
        model.test_(test_loader, epoch)

    torch.save(model, 'student_output.pkl')
