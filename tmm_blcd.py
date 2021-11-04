# -*- coding: utf-8 -*-
"""tmm-blcd (2).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Yg8nRUHz7OZQjZhwpD7bkVlLwa0Ler4Z
"""
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from math import pi, sqrt
import torchvision.datasets as datasets
from PIL import Image
from matplotlib import pyplot as plt
from time import time
import numpy as np
import operator

t = 0.0025
m = 0.6
K = 16  # 领域数量
epochs = 80
weight_decay = 0.0001
momentum = 0.9
lr = 10  # 80个epochs中对数下降至0.001
batch_size = 256
input_size = (32, 32, 1)
saved = False
use_cuda = False

means = {'notredame': 0.4854, 'yosemite': 0.4844, 'liberty': 0.4437,
         'notredame_harris': 0.4854, 'yosemite_harris': 0.4844, 'liberty_harris': 0.4437}
stds = {'notredame': 0.1864, 'yosemite': 0.1818, 'liberty': 0.2019,
        'notredame_harris': 0.1864, 'yosemite_harris': 0.1818, 'liberty_harris': 0.2019}


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, gamma=1):
        super(BatchNorm, self).__init__(num_features)
        self.affine = False
        if gamma:
            self.weight = nn.Parameter(torch.ones(num_features))
        else:
            self.weight = nn.Parameter(torch.full_like(torch.ones(num_features), sqrt(2 / pi)))
        self.moving_mean = torch.zeros(num_features)
        self.moving_var = torch.zeros(num_features)


class BLCD_Model(nn.Module):
    def __init__(self):
        super(BLCD_Model, self).__init__()

        self.net = nn.Sequential(
            # 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            BatchNorm(32),
            nn.ReLU(),
            # 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            BatchNorm(64),
            nn.ReLU(),
            # 3
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            BatchNorm(64),
            nn.ReLU(),
            # 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            BatchNorm(128),
            nn.ReLU(),
            # 5
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            BatchNorm(128),
            nn.ReLU(),
            # 6
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            BatchNorm(128),
            nn.ReLU(),
            # 7
            nn.Conv2d(128, 256, kernel_size=8),
            BatchNorm(256, 0)
        )

    def forward(self, x):
        out = self.net(x)
        return out

    def _initialize_weights(self):
        # print(self.modules())
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                # print(m.weight.data.type())
                # input()
                # m.weight.data.fill_(1.0)
                nn.init.xavier_uniform_(m.weight, gain=1)


def dis(yi, yj):
    out = yi / torch.sqrt(torch.sum(yi**2, dim=-1)).unsqueeze(-1).expand(256, 256) - yj / torch.sqrt(torch.sum(yj**2, dim=-1)).unsqueeze(-1).expand(256, 256)
    return 0.5*torch.sqrt(torch.sum(out**2, dim=-1))

class BLCD_Loss(nn.Module):
    def __init__(self, t, m, K):
        super(BLCD_Loss, self).__init__()
        self.t = t
        self.m = m
        self.K = K

    def forward(self, yi, yi_t):
        # yi归一化
        yi = yi / torch.sqrt(torch.sum(yi ** 2, dim=-1) + 1e-12).unsqueeze(-1).expand(256, 256)
        # 计算出yi中每个向量与其他向量的dis -> dis_yii
        yii = yi.unsqueeze(1) - yi.unsqueeze(0)
        dis_yii = 0.5 * torch.sqrt(torch.sum(yii ** 2, dim=-1) + 1e-12)

        # yi_t归一化
        yi_t = yi_t / torch.sqrt(torch.sum(yi_t ** 2, dim=-1) + 1e-12).unsqueeze(-1).expand(256, 256)

        # 离yi最近的K个特征 [256，16]
        topk_value_yi, topk_index_yi = torch.topk(dis_yii, k=self.K + 1, dim=1, largest=False)

        # yij：离yi最近的特征
        yij_value, yij_index = torch.topk(dis_yii, k=2, dim=1, largest=False)

        # 把topk_index_yi中的[256,16]构建成yj: [256,16,256]
        slice = []
        for i in range(topk_index_yi.shape[0]):
            slice.append(torch.index_select(yi, 0, topk_index_yi[i][1:]))
            # yi_topk = torch.stack([yi_topk, torch.index_select(yi, 0, topk_index_yi[i]).unsqueeze(0)], dim=0)
        yj = torch.stack(slice, dim=0)

        # [256,256]-[256,16,256] -> [256,16,256]
        # dis(yi,yj) & dis(yi+,yj)
        dis_yij = 0.5 * torch.sqrt(torch.sum((yi.unsqueeze(1) - yj) ** 2, dim=-1) + 1e-12)
        dis_yi_tj = 0.5 * torch.sqrt(torch.sum((yi_t.unsqueeze(1) - yj) ** 2, dim=-1) + 1e-12)

        # e1
        out1 = (dis_yij - dis_yi_tj) ** 2- t
        e1 = torch.sum(out1)

        # e2
        dis_yi_yi_t = 0.5 * torch.sqrt(torch.sum((yi - yi_t) ** 2, dim=-1) + 1e-12)
        yij_value, yij_index = torch.topk(dis_yii, k=2, dim=-1, largest=False)
        _, yij_value = torch.chunk(yij_value, 2, dim=-1)
        yij_value = torch.squeeze(yij_value)
        # print(f'dis_yi yi+: {dis_yi_yi_t.shape}, yij_value: {yij_value.shape}')
        out = dis_yi_yi_t + self.m - yij_value
        # print(f'out: {out.shape}')
        e2 = torch.sum(torch.where(out > 0, out, torch.zeros(out.shape).cuda()))

        # e1 = DLC_Loss(dis_yii, dis_yij_t, self.t, self.K)
        # e2 = self_Distinct(yi, dis_yii, dis_yii_t, self.m)
        # print(f'e1: {e1}, e2: {e2}')
        return e1 + e2


def DLC_Loss(dis_yii, dis_yij_t, t, K):
    topk_value_yi, topk_index_yi = torch.topk(dis_yii, k=K + 1, dim=-1, largest=False)
    topk_value_yi_t, topk_index_yi_t = torch.topk(dis_yij_t, k=K + 1, dim=-1, largest=False)
    out = (topk_value_yi - topk_value_yi_t) ** 2 - t
    out = torch.mean(torch.sum(torch.where(out > 0, out, torch.zeros(out.shape)), dim=-1))
    return out


def self_Distinct(yi, dis_yii, dis_yii_t, m):
    yij_value, yij_index = torch.topk(dis_yii, k=2, dim=-1, largest=False)
    _, yij_value = torch.chunk(yij_value, 2, dim=-1)
    out = yij_value + m - dis_yii_t
    out = torch.mean(torch.sum(torch.where(out > 0, out, torch.zeros(out.shape)), dim=-1))
    return out


def train(train_loader, model, criterion, optimizer):
    model.train()
    train_loss = []

    for i, (input1, input2) in enumerate(train_loader):
        if use_cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
        # print(f'i1: {input1.shape}, i2: {input2.shape}')
        output1 = model(input1)
        output2 = model(input2)
        # print(f'o1: {output1.shape}, o2: {output2.shape}')
        output1 = torch.squeeze(output1)
        output2 = torch.squeeze(output2)

        # print(f'o1: {output1}, o2: {output2}')

        loss = criterion(output1, output2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    return np.mean(train_loss)


class ImagePair(datasets.PhotoTour):
    """name: notredame, yosemite, liberty"""

    def __init__(self, root, name, train=True, transform=None, download=False):
        super(ImagePair, self).__init__(root, name, train, transform, download)

    def __getitem__(self, index):
        if self.train:
            data = self.data[index].type(torch.float32)
            data -= torch.mean(data)
            data /= stds[self.name]
            train_data = Image.fromarray(data.numpy()).resize((32, 32))
            if self.transform is not None:
                trans = transforms.Compose([transforms.ToTensor()])
                data_trans = self.transform(train_data)
                train_data = trans(train_data)
            return train_data, data_trans
        m = self.matches[index]
        data1, data2 = self.data[m[0]], self.data[m[1]]
        data1 = Image.fromarray(data1.numpy()).resize((32, 32))
        data2 = Image.fromarray(data2.numpy()).resize((32, 32))
        if self.transform is not None:
            data1 = self.transform(data1)
            data2 = self.transform(data2)
        return data1, data2, m[2]


def get_fpr_at_95_recall(labels, scores):
    recall_point = 0.95
    # Sort label-score tuples by the score in descending order.
    temp = zip(labels, scores)
    # operator.itemgetter(1)按照第二个元素的次序对元组进行排序，reverse=True是逆序，即按照从大到小的顺序排列
    # sorted_scores.sort(key=operator.itemgetter(1), reverse=True)
    sorted_scores = sorted(temp, key=operator.itemgetter(1), reverse=False)

    # Compute error rate
    # n_match表示测试集正样本数目
    n_match = sum(1 for x in sorted_scores if x[0] == 1)
    n_thresh = recall_point * n_match
    tp = 0
    count = 0
    for label, score in sorted_scores:
        count += 1
        if label == 1:
            tp += 1
        if tp >= n_thresh:
            break
    return float(count - tp) / (len(sorted_scores) - n_match)


def FPR95(model, test_loader):
    fpr95 = []

    model.eval()
    with torch.no_grad():
        for i, (input1, input2, label) in enumerate(test_loader):
            if use_cuda:
                input1 = input1.cuda()
                input2 = input2.cuda()
            output1 = model(input1)
            output2 = model(input2)

            output1 = torch.squeeze(output1)
            output2 = torch.squeeze(output2)

            output1 = torch.sign(output1)
            output2 = torch.sign(output2)

            score = dis(output1, output2)

            fpr95.append(get_fpr_at_95_recall(label, score))

    return np.mean(fpr95)


train_trans = transforms.Compose([
    transforms.RandomAffine(90, (0.25, 0.25), (0.8, 1.2), (-20, 20)),
    transforms.ToTensor()
])
test_trans = transforms.Compose([transforms.ToTensor()])

train_data = ImagePair(r'datasets\notredame', 'notredame', train=True, transform=train_trans, download=False)

train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=False, drop_last=True)

test_data = ImagePair(r'datasets\liberty', 'liberty', train=False, transform=test_trans, download=False)

test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, drop_last=True)

model = BLCD_Model()
model._initialize_weights()
if use_cuda:
    model.cuda()
if saved:
    model.load_state_dict(torch.load('/content/drive/MyDrive/TMM_BLCD/model/BLCD_model_40epochs.pth'))

# torch.save(model, '/content/drive/MyDrive/TMM_BLCD/BLCD_model.pth')

criterion = BLCD_Loss(t, m, K)

optimizer = torch.optim.SGD(model.parameters(), lr, momentum, weight_decay)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

best_loss = 10000.0

loss = []
fpr95 = []
print('start training')
for epoch in range(epochs):
    start = time()
    train_loss = train(train_loader, model, criterion, optimizer)
    fpr = FPR95(model, test_loader)
    scheduler.step()
    print(f'epoch: {epoch + 1}, train_loss: {train_loss} , FPR95: {fpr}, cost time: {time() - start}')
    loss.append(train_loss)
    fpr95.append(fpr)
    if train_loss < best_loss:
        best_loss = train_loss
        if saved:
            torch.save(model.state_dict(), f'/content/drive/MyDrive/TMM_BLCD/model/BLCD_model_new_{epoch + 1}.pth')
        else:
            torch.save(model.state_dict(), '/content/drive/MyDrive/TMM_BLCD/model/BLCD_model.pth')

plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot([i for i in range(1, len(loss) + 1)], loss)

plt.show()

plt.title('FPR95')
plt.xlabel('epoch')
plt.ylabel('FPR95')
plt.plot([i for i in range(1, len(fpr95) + 1)], fpr95)

plt.show()

with open('loss.txt', 'a') as f:
    for l in loss:
        f.write(str(l) + ' ')

with open('fpr95.txt', 'a') as f:
    for l in fpr95:
        f.write(str(l) + ' ')