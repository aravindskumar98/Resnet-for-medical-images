'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
from utils import progress_bar, get_lr, recycle
from efficientnet_pytorch import EfficientNet

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.1, type=float, help='lr decay')
parser.add_argument('--wd', default=1e-6, type=float, help='weights decay')
parser.add_argument('--ne', default=30, type=int, help='number of epochs')
parser.add_argument('--nsc', default=10, type=int, help='number of step for a lr')
parser.add_argument('--batch_split', default=1, type=int, help='spliting factor for the batch')
parser.add_argument('--batch', default=32, type=int, help='size of the batch')
parser.add_argument('--alpha', default=0.1, type=float,
                    help='mixup interpolation coefficient (default: 1)')

args = parser.parse_args()

#To get reproductible experiment
torch.manual_seed(0)
np.random.seed(0)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device : ",device)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=160,scale=(0.6,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


transform_test = transforms.Compose([
    transforms.Resize(200),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_and_val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
from torch.utils.data import Subset

indices = np.arange(len(test_and_val_set))
val_set = Subset(test_and_val_set, indices[:5000])
test_set = Subset(test_and_val_set, indices[5000:])

micro_batch_size = args.batch // args.batch_split


criterion = nn.CrossEntropyLoss()

def mixup_data(x, y, alpha=1.0,lam=1.0,count=0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if count == 0:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Training
def train(epoch,trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    count = 0
    lam = 1.0
    optimizer.zero_grad()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if count == args.batch_split:
            optimizer.step()
            optimizer.zero_grad()
            count = 0
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,args.alpha,lam,count)
        outputs = net(inputs)
        loss =  mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss = loss / args.batch_split
        loss.backward()
        count +=1
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(testloader,namesave):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, namesave)

def validate(testloader):
    
    net.eval()
    
    counter = 0
    correctly_predicted_counter = 0
    predicted_classes_list = []
    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            _, predicted_classes = outputs.max(1)

            counter += targets.size(0)
            correctly_predicted_counter += (predicted_classes == targets).sum().item()
    
    
    # predicted_classes_list = []

    # with torch.no_grad():
    #     for batch_idx, (inputs, targets) in enumerate(test_set_loader):
    #         inputs, targets = inputs.cuda(), targets.cuda()
    #         outputs = net(inputs)
            # _, predicted_classes = outputs.max(1)
            predicted_classes_list += list(map(lambda x: str(x), predicted_classes.cpu().detach().numpy().tolist()))

    predicted_classes_list = list(enumerate(predicted_classes_list))
    accuracy = float(correctly_predicted_counter) / counter

    net.train()
    
    return accuracy, predicted_classes_list
        
import pandas as pd
testloader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False, num_workers=1)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=micro_batch_size, shuffle=True, num_workers=1)        
best_accuracy = 0
net = EfficientNet.from_pretrained('efficientnet-b4', num_classes=10)
net = net.to(device)
namesave='./checkpoint/ckpt'
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
lr_sc = lr_scheduler.StepLR(optimizer, step_size=args.nsc, gamma=args.gamma)
for epoch in range(0, args.ne):
    train(epoch,trainloader)
    lr_sc.step()
    current_accuracy, predicted_classes_list = validate(testloader)
    if current_accuracy > best_accuracy:
        print(current_accuracy)
        best_accuracy = current_accuracy
        torch.save(net.state_dict(), 'resnet_20_cifar10_kaggle.pth')
        # Saving best predictions so far
        submission_df = pd.DataFrame(predicted_classes_list, columns = ['Id', 'Category'])
        submission_df = submission_df['Category']
        submission_df = submission_df.replace(0)
        submission_df = pd.DataFrame(submission_df)
        submission_df['Id'] = submission_df.index
        submission_df = submission_df[['Id', 'Category']]
        submission_df.to_csv('cifar_10_best_submission.csv', index=False)

print("Test accuracy : ")
test(testloader,namesave)
    