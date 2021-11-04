from torchvision import datasets
from PIL import Image
import numpy as np
from blcd_best import Metrix_afiine

class CIFAR10Pair(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, download=False):
        super(CIFAR10Pair, self).__init__(root, train, transform, download)

    def __getitem__(self, index):
        if self.train:
            image = self.data[index]
            img1 = Image.fromarray(image)
            img2 = Image.fromarray(image)
            if self.transform is not None:
                a,b,t = np.random.randint(-20,20), np.random.randint(-20,20), 1
                scale = np.random.uniform(0.8,1.2)
                translation_x = np.random.randint(-8,8)
                translation_y = np.random.randint(-8,8)
                img_2 = Metrix_afiine(a,b,t,translation_x,translation_y,scale)
                img_1 = self.transform(img1)
                img_2 = self.transform(img2)
                img_1 = (img_1 - img_1.mean()) / img_1.std()
                img_2 = (img_2 - img_2.mean()) / img_2.std()
            return img_1, img_2
        index2 = np.random.randint(0,60000)
        img1, target1 = self.data[index], self.targets[index]
        img2, target2 = self.data[index2], self.targets[index2]
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        # print(img1, target1)
        if self.transform is not None:
            img_1 = self.transform(img1)
            img_2 = self.transform(img2)
            img_1 = (img_1 - img_1.mean()) / img_1.std()
            img_2 = (img_2 - img_2.mean()) / img_2.std()
            if target1==target2:
                label = 1
            else:
              label = 0
        return img_1, img_2, label
