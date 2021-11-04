# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from math import pi,sqrt 
import torchvision.datasets as datasets
from PIL import Image
from matplotlib import pyplot as plt
from time import time
import numpy as np
import operator
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", type=int, default=64)
parser.add_argument("-m", type=float, default=0.6)
args = parser.parse_args()


t = 0.0025
m = args.m
b = args.b
print(b)
K = 16     # 领域数量
epochs = 80
weight_decay = 0.0001
momentum = 0.9
lr = 10    #80个epochs中对数下降至0.001
batch_size = 256
#input_size = (32, 32, 1)
saved = False
use_cuda = True
start_epoch = 0
train_mode = True
# subdataset = 'y_to_n'

if f'cifar10_{b}' not in os.listdir('BLCD/models'):
    os.mkdir(f'BLCD/models/cifar10_{b}')

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, sign=True):
        super(BatchNorm, self).__init__(num_features)
        if sign:
          self.weight = nn.Parameter(torch.ones(num_features))
        else:
          self.weight = nn.Parameter(torch.full_like(torch.ones(num_features), sqrt(2 / pi)))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.weight.requires_grad = False
        self.bias.requires_grad = False

class BLCD_Model(nn.Module):
    def __init__(self):
        super(BLCD_Model, self).__init__()

        self.net = nn.Sequential(
            #1
            nn.Conv2d(3, 32, 3, padding=1),
            # nn.BatchNorm2d(32,affine=False),
            BatchNorm(32),
            nn.ReLU(),
            #2
            nn.Conv2d(32, 64, 3, padding=1),
            #nn.BatchNorm2d(64,affine=False),
            BatchNorm(64),
            nn.ReLU(),
            #3
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            #nn.BatchNorm2d(64,affine=False),
            BatchNorm(64),
            nn.ReLU(),
            #4
            nn.Conv2d(64, 128, 3, padding=1),
            #nn.BatchNorm2d(128,affine=False),
            BatchNorm(128),
            nn.ReLU(),
            #5
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            #nn.BatchNorm2d(128,affine=False),
            BatchNorm(128),
            nn.ReLU(),
            #6
            nn.Conv2d(128, 128, 3, padding=1),
            #nn.BatchNorm2d(128,affine=False),
            BatchNorm(128),
            nn.ReLU(),
            #7
            nn.Conv2d(128, 256, 8),
            BatchNorm(256, False)
            )
        self.fc = nn.Linear(256, b, bias=True)
        
    def forward(self, x):
        x = self.net(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        return x
    
    
    def _initialize_weights(self):
        # print(self.modules())
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                # print(m.weight.data.type())
                # input()
                # m.weight.data.fill_(1.0)
                nn.init.xavier_uniform_(m.weight, gain=1)


class BLCD_Loss(nn.Module):
    def __init__(self, t, m, K):
        super(BLCD_Loss, self).__init__()
        self.t = t
        self.m = m
        self.K = K
    
    def forward(self, yi, yi_t):
        # yi = torch.squeeze(yi)
        # # print(f'yi: {yi}')
        # yi_t = torch.squeeze(yi_t)
        # yi汉明距离
        yi_l2 = yi / torch.sqrt(torch.sum(yi**2, dim=-1) + 1e-12).unsqueeze(-1).expand(yi.shape)
        # 计算出yi中每个向量与其他向量的dis -> dis_yii
        # yii_l2 = yi.unsqueeze(1) - yi.unsqueeze(0)
        yii_l2 = yi_l2.unsqueeze(1) - yi_l2.unsqueeze(0)
        # dis_yii = torch.sqrt(torch.sum(yii**2, dim=-1) + 1e-12)
        dis_yii_l2 = torch.sqrt(torch.sum(yii_l2**2, dim=-1) + 1e-12)

        # 离yi最近的K个特征 [256，16]
        # topk_value_yi, topk_index_yi = torch.topk(dis_yii, k=self.K + 1, dim=-1, largest=False)
        topk_value_yi, topk_index_yi = torch.topk(dis_yii_l2, k=self.K + 1, dim=-1, largest=False)

        # _, dis_yij = torch.split(topk_value_yi, [1,16], dim=-1)
        # print(f'yi: {topk_index_yi}, yi_l2: {topk_index_yi_l2}')
        yi = yi / torch.sqrt(torch.sum(yi**2, dim=-1) + 1e-12).unsqueeze(-1).expand(yi.shape)
        # yi_t汉明距离
        yi_t = yi_t / torch.sqrt(torch.sum(yi_t**2, dim=-1) + 1e-12).unsqueeze(-1).expand(yi_t.shape)
        # 把topk_index_yi中的[256,16]构建成yj: [256,16,256]
        # print(f'index: {topk_index_yi}, value: {topk_value_yi}')
        slice_k = []
        slice_1 = []
        for i in range(topk_index_yi.shape[0]):
            # 离yi最近的k个特征
            topk = torch.index_select(yi, 0, topk_index_yi[i][1:])
            # 离yi最近的特征
            top1 = torch.index_select(yi, 0, topk_index_yi[i][1:2])
            # print(f'topk: {topk.shape, topk}')
            slice_k.append(topk)
            slice_1.append(top1)
            # yi_topk = torch.stack([yi_topk, torch.index_select(yi, 0, topk_index_yi[i]).unsqueeze(0)], dim=0)
        yj = torch.stack(slice_k, dim=0)
        yij = torch.stack(slice_1, dim=0)
        yij = torch.squeeze(yij)

        # yj = yj / torch.sqrt(torch.sum(yj**2, dim=-1) + 1e-12).unsqueeze(-1).expand(yj.shape)

        # [256,1,256]-[256,16,256] -> [256,16,256]
        # dis(yi,yj) & dis(yi+,yj)
        dis_yij = 0.5 * torch.sqrt(torch.sum((yi.unsqueeze(1) - yj)**2, dim=-1) + 1e-12)
        dis_yi_tj = 0.5 * torch.sqrt(torch.sum((yi_t.unsqueeze(1) - yj)**2, dim=-1) + 1e-12)
        # print(f'dis_yij: {dis_yij.shape}, dis_yi_tj: {dis_yi_tj.shape}')
        # e1
        # print(f'dis_yij: {dis_yij.shape}, dis_yi_tj: {dis_yi_tj.shape}')
        out1 = (dis_yij - dis_yi_tj)**2 - self.t
        e1 = torch.sum(torch.where(out1 > 0, out1, torch.zeros(out1.shape).cuda()))

        # e2
        dis_yi_yi_t = 0.5 * torch.sqrt(torch.sum((yi - yi_t) ** 2, dim=-1) + 1e-12)
        dis_yi_yij = 0.5 * torch.sqrt(torch.sum((yi - yij)**2, dim=-1) + 1e-12)
        # _, yij_value = torch.chunk(yij_value, 2, dim=-1)
        # print(f'yij_value: {yij_value.shape, yij_value}')
        # yij_value = torch.squeeze(yij_value)
        # print(f'dis_yi yi+: {dis_yi_yi_t}, dis_yi_yij: {dis_yi_yij}')
        out2 = dis_yi_yi_t + self.m - dis_yi_yij
        # print(f'out2: {out2, out2.shape}')
        # print(f'out2: {torch.where(out2 > 0, out2, torch.zeros(out2.shape).cuda())}')
        e2 = torch.sum(torch.where(out2 > 0, out2, torch.zeros(out2.shape).cuda()))
        # print(f'e1: {e1}, e2: {e2}')
        return e1 + e2, e1, e2
      
def train(train_loader, model, criterion, optimizer):
    
    train_loss = []
    train_loss1 = []
    train_loss2 = []

    for i, (input1, input2) in enumerate(train_loader):
        model.train()
        if use_cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
        output1 = model(input1)
        output2 = model(input2)

        loss,loss1,loss2= criterion(output1, output2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_loss1.append(loss1.item())
        train_loss2.append(loss2.item())

    return np.mean(train_loss),np.mean(train_loss1),np.mean(train_loss2)

def dis(yi, yj):
    yi = yi / torch.sqrt(torch.sum(yi**2, dim=-1)).unsqueeze(-1).expand(yi.shape)
    yj = yj / torch.sqrt(torch.sum(yj**2, dim=-1)).unsqueeze(-1).expand(yj.shape)
    return torch.sqrt(torch.sum((yi-yj)**2, dim=-1))

def ErrorRateAt95Recall1(labels, scores):
    recall_point = 0.945
    labels = np.asarray(labels.cpu())
    scores = np.asarray(scores.cpu())
    # Sort label-score tuples by the score in descending order.
    indices = np.argsort(scores)   #升序排列
    sorted_labels = labels[indices]
    sorted_scores = scores[indices]
    # print(f'sort score: {sorted_scores}')
    n_match = sum(sorted_labels)
    n_thresh = recall_point * n_match
    thresh_index = np.argmax(np.cumsum(sorted_labels) >= n_thresh)
    FP = np.sum(sorted_labels[:thresh_index] == 0)
    TN = np.sum(sorted_labels[thresh_index:] == 0)
    return float(FP) / float(FP + TN)

def get_fpr_at_95_recall(labels, scores):
    recall_point = 0.945
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
    FPR95_y = []
    FPR95_b = []
    
    model.eval()
    with torch.no_grad():
        for i, (input1, input2, label) in enumerate(test_loader):
            if use_cuda:
                input1 = input1.cuda()
                input2 = input2.cuda()
            output1 = model(input1)
            output2 = model(input2)
            # print(torch.mean(input1), output1)
            output1 = torch.squeeze(output1)
            output2 = torch.squeeze(output2)
            score_y = dis(output1, output2)
            # print()
            # plt.hist(output1.cpu().reshape(256*256), 300)
            # plt.show()
            # break

            FPR95_y.append(get_fpr_at_95_recall(label, score_y))
            #FPR95_y.append(ErrorRateAt95Recall1(label, score_y))
            output1 = torch.sign(output1)
            output2 = torch.sign(output2)
            # print(f'o1: {output1}, o2: {output2}, mean: {torch.mean(torch.sign(output1[0])),torch.mean(torch.sign(output2[0]))}')
            score_b = dis(output1, output2)
            # print(f'socre: {score_b}')
            FPR95_b.append(get_fpr_at_95_recall(label, score_b))
            #FPR95_b.append(ErrorRateAt95Recall1(label, score_b))

    return np.mean(FPR95_y), np.mean(FPR95_b)

def rad(x):
    return x * np.pi / 180
  
def Metrix_afiine(img,a,b,t,translation_x, translation_y, scale):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)

    # 面内/面外旋转 + 倾斜
    R1 = np.array([[np.cos(rad(a)), -np.sin(rad(a))], [np.sin(rad(a)), np.cos(rad(a))]], np.float32)
    T = np.array([[t, 0], [0, 1]], np.float32)
    R2 = np.array([[np.cos(rad(b)), -np.sin(rad(b))], [np.sin(rad(b)), np.cos(rad(b))]], np.float32)

    A = scale*R1.dot(T).dot(R2)

    # M = np.zeros((2,3),dtype=np.float32)
    #
    # M[0:2, 0:2] += A
    pts1 = np.float32([[0, img.shape[1] - 1], [img.shape[0] - 1, 0], [img.shape[0] - 1, img.shape[1] - 1]])
    # pts1 = np.float32([[0,0],[0,img.shape[1]-1],[img.shape[0]-1,0],[img.shape[0]-1, img.shape[1]-1]])
    pts2 = pts1.dot(A)

    M = cv2.getAffineTransform(pts1, pts2)
    #M = cv2.getPerspectiveTransform(pts1, pts2)
    # Metrix += translation_scale

    # M[0,2] += translation_x
    # M[1,2] += translation_y
    # print(Metrix)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX + translation_x
    M[1, 2] += (nH / 2) - cY + translation_y
    #Metrix += translation_scale
    trans = cv2.warpAffine(img, M, (nW, nH))

    #trans = cv2.warpPerspective(img, M, (nW, nH))
    
    #trans = cv2.resize(trans,(32,32))
    trans = affine(trans, t, t)

    return trans

def affine(img, dx=0, dy=0):
    # get shape
    (H, W, C) = img.shape[:]

    # Affine hyper parameters
    a = 1.
    b = dx / H
    c = dy / W
    d = 1.
    tx = 0.
    ty = 0.

    # prepare temporary
    _img = np.zeros((H+2, W+2, C), dtype=np.float32)

    # insert image to center of temporary
    _img[1:H+1, 1:W+1, :] = img

    # prepare affine image temporary
    H_new = np.ceil(dy + H).astype(np.int32)
    W_new = np.ceil(dx + W).astype(np.int32)
    out = np.zeros((H_new, W_new, C), dtype=np.float32)

    # preprare assigned index
    x_new = np.tile(np.arange(W_new), (H_new, 1))
    y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

    # prepare inverse matrix for affine
    adbc = a * d - b * c
    x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int32) - tx + 1
    y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int32) - ty + 1

    x = np.minimum(np.maximum(x, 0), W+1).astype(np.int32)
    y = np.minimum(np.maximum(y, 0), H+1).astype(np.int32)

    # assign value from original to affine image
    out[y_new, x_new, :] = _img[y, x, :]
    out = out.astype(np.uint8)
    out = cv2.resize(out,(32,32))
    return out

class CIFAR10Pair(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, download=False):
        super(CIFAR10Pair, self).__init__(root, train, transform, download)

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        if self.train:
            # img1 = Image.fromarray(image)
            # img2 = Image.fromarray(image)
            if self.transform is not None:
                img_1 = self.transform(image)
                a,b,t = np.random.randint(-20,20), np.random.randint(-20,20), 1
                scale = np.random.uniform(0.8,1.2)
                translation_x = np.random.randint(-4,4)
                translation_y = np.random.randint(-4,4)
                img_2 = Metrix_afiine(image,a,b,t,translation_x,translation_y,scale)
                img_2 = self.transform(img_2)
                img_1 = (img_1 - img_1.mean()) / img_1.std()
                img_2 = (img_2 - img_2.mean()) / img_2.std()
            return img_1, img_2
        # print(img1, target1)
        if self.transform is not None:
            img_1 = self.transform(image)
            img_1 = (img_1 - img_1.mean()) / img_1.std()
        return img_1, target



train_trans = transforms.Compose([
    #transforms.RandomAffine(degrees=0, translate=(0.25, 0.25), scale=(0.8,1.2), shear=(-20,20,-20,20)),
    #transforms.Resize([32,32]),
    transforms.ToTensor()
])
test_trans = transforms.Compose([
    #transforms.Resize([32,32]), 
    transforms.ToTensor()
    ])

cifar10_trans = transforms.Compose([
    transforms.ToTensor()
    ])


# train_data = ImagePair(f'datasets/{dataset_train}', dataset_train, train=True, transform=train_trans, download=False)

# train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, drop_last=True)

# test_data = ImagePair(f'datasets/{dataset_test}', dataset_test, train=False, transform=test_trans, download=False)

# test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True, drop_last=True)

cifar10_train = CIFAR10Pair('datasets/cifar10/', train=True, transform=cifar10_trans, download=False)
train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size, shuffle=True)

memory_data = CIFAR10Pair('datasets/cifar10/', train=False, transform=cifar10_trans, download=False)
memory_loader = torch.utils.data.DataLoader(memory_data, batch_size, shuffle=False)

cifar10_test = CIFAR10Pair('datasets/cifar10/', train=False, transform=cifar10_trans, download=False)
test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size, shuffle=False)

model = BLCD_Model()
model._initialize_weights()
if not train_mode:
    model.load_state_dict(torch.load(f'BLCD/models/cifar10_{b}/BLCD_restart_model_cifar10_{b}_4.pth'))
model.cuda()

criterion = BLCD_Loss(t, m, K)

# optimizer = torch.optim.SGD(model.parameters(), lr, momentum, weight_decay)
optimizer=torch.optim.Adam(model.parameters(),lr=0.001,betas=(0.9, 0.99),eps=1e-08,weight_decay=0)

# optimizer = torch.optim.Adam(model.parameters(), lr, betas=(momentum, momentum), weight_decay=weight_decay)

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.1)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.001, 3, verbose=True, min_lr=0.00001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=0.00001)
  
if train_mode:
    print('start training')
    for epoch in range(start_epoch, epochs):
        start = time()
        # e,e1,e2 = 0,0,0
        e,e1,e2= train(train_loader, model, criterion, optimizer)
        # fpr_y,fpr_b = FPR95(model, test_loader)
        #adjust_lr(optimizer, epoch, lr)
        scheduler.step()
        result = f'epoch: {epoch+1} , loss: {e} ,loss1: {e1} ,loss2: {e2}, cost time: {time() - start}'
        print(result)
        # with open(f'BLCD/log/record_cifar10_{bit}.txt', 'a') as f:
        #     f.write(result + '\n')
        torch.save(model.state_dict(), f'BLCD/models/cifar10_{b}/BLCD_restart_model_cifar10_{b}_{epoch+1}.pth')

def bin_sign(x):
    return 0.5 * (torch.sign(x) + 1)

def predict(memory_loader, test_loader, model, c=10):
    model.eval()
    k = 1000
    mAP = []
    with torch.no_grad():
        feature_matrix = []
        for i, (input1, input2) in enumerate(memory_loader):
            if use_cuda:
                input1 = input1.cuda()
            output1 = model(input1)
            output1 = torch.squeeze(output1)
            # feature_matrix.append(bin_sign(output1))
            feature_matrix.append(output1)
        feature_matrix = torch.t(torch.cat(feature_matrix, dim=0))
        feature_matrix = feature_matrix / torch.unsqueeze(torch.sqrt(torch.sum(feature_matrix**2, dim=0) + 10e-9), dim=0)
        # feature_matrix = feature_matrix / torch.unsqueeze(torch.sqrt(torch.sum(feature_matrix, dim=0) + 10e-9), dim=0)
        feature_labels = torch.tensor(memory_loader.dataset.targets, device=torch.device('cuda' if use_cuda else 'cpu'))
        top1_num, top5_num, tot_num = 0, 0, 0
        for i, (input1, labels) in enumerate(test_loader):
            if use_cuda:
                input1 = input1.cuda()
                labels = labels.cuda()
            feature = model(input1)
            feature = torch.squeeze(feature)
            # sim_matrix = torch.mm(bin_sign(feature), feature_matrix)
            sim_matrix = torch.mm(feature, feature_matrix)
            # 获取训练集中相似度最高的前K个值sim_value，以及下标sim_index
            sim_value, sim_index = torch.topk(sim_matrix, k=k, dim=1)
            sim_labels = torch.gather(feature_labels.expand(input1.shape[0], -1), dim=1, index=sim_index)
            # top-1 top-5
            one_hot = torch.zeros(input1.shape[0] * k, c, device=sim_labels.device).scatter_(dim=1, index=sim_labels.view(-1, 1), value=1)
            # print(one_hot)
            # b * k * c -----> b * c weight per class
            scores = torch.sum(one_hot.view(input1.shape[0], k, -1) * torch.unsqueeze(sim_value, dim=-1), dim=1)
            # print(scores)
            # b * c
            predict = torch.argsort(scores, dim=1, descending=True)
            tot_num += input1.shape[0]
            top1_num += (torch.sum((predict[:, :1] == torch.unsqueeze(labels, dim=-1)).any(dim=-1))).item()
            top5_num += (torch.sum((predict[:, :5] == torch.unsqueeze(labels, dim=-1)).any(dim=-1))).item()
            # mAP
            # positive_labels = torch.where((sim_labels == torch.unsqueeze(labels, dim=-1)) > 0, 1, 0)
            positive_labels = (sim_labels == torch.unsqueeze(labels, dim=-1))
            result = torch.zeros(positive_labels.size())
            positive_nums = torch.zeros(positive_labels.size()[0])
            for i in range(positive_labels.size()[0]):
                index = 0
                positive_num = 0
                for j in range(positive_labels.size()[1]):
                    index += 1
                    if positive_labels[i][j]:
                        positive_num += 1
                        result[i][j] = positive_num / index
                if positive_num == 0:
                    positive_num = 1
                positive_nums[i] = positive_num
            AP = torch.sum(result, dim=1)
            AP = AP / positive_nums
            mAP.append(torch.mean(AP).item())
    return np.mean(mAP), top1_num / tot_num, top5_num / tot_num

mAP, top1, top5 = predict(memory_loader, test_loader, model)
print(mAP, top1, top5)