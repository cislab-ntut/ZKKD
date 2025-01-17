'''
teacher model training

exec: 
$ python teacher_model_training.py

accuracy: 0.9760
training time: 2 minutes

dependency:
$ pip install torch torchvision tqdm

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
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from torchvision import transforms

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

def teacher_train(device, train_loader, test_loader):
    print('--------------teachermodel start--------------')
    model = TeacherModel()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 6
    for epoch in range(epochs):
        model.train()

        for data, target in tqdm(train_loader):
            data = data.to(device)
            target = target.to(device)
            preds = model(data)
            loss = criterion(preds, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        num_correct = 0
        num_samples = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                preds = model(x)
                predictions = preds.max(1).indices
                num_correct += (predictions.eq(y)).sum().item()
                num_samples += predictions.size(0)
            acc = num_correct / num_samples

        model.train()
        print('Epoch:{}\t Acc:{:.4f}'.format(epoch + 1, acc))
    torch.save(model, 'teacher.pkl')
    print('--------------teachermodel end--------------')


if __name__ == '__main__':
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    torch.backends.cudnn.benchmark = True

    X_train = torchvision.datasets.MNIST(
        root="dataset/",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    X_test = torchvision.datasets.MNIST(
        root="dataset/",
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    train_loader = DataLoader(dataset=X_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=X_test, batch_size=32, shuffle=False)

    #從頭訓練教師模型，並預測
    teacher_train(device, train_loader, test_loader)