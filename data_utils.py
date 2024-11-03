import numpy as np
import os
import cv2
import time
import random
from pathlib import Path
import pandas as pd

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor

class Data:

    def __init__(self, root, train_n, test_n, seed=0, transform=None):
        self.transform = transform
        self.root = Path(root)
        self.train_n = train_n
        self.test_n = test_n
        self.seed = seed

        train_path = f'./data/train{self.train_n}.csv'
        test_path = f'./data/test{self.test_n}.csv'

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            self.data_split()

        self.train_data = MiniImageNet(self.load_csv(train_path), transform=self.transform)
        self.test_data = MiniImageNet(self.load_csv(test_path), transform=self.transform)
  
    def load_csv(self, path):
        f = open(path, 'r')
        lines = f.readlines()
        data = [(str(file), int(label)) for file, label in [line.strip().split(',') for line in lines[1:]]]
        return data

    def data_split(self):
        random.seed(self.seed)
        train_data = []
        test_data = []
        for idx, folder in enumerate(self.root.iterdir()):
            subfiles = list(folder.iterdir())
            random.shuffle(subfiles)
            train_data.extend([(str(file), idx) for file in subfiles[:self.train_n]])
            test_data.extend([(str(file), idx) for file in subfiles[self.train_n:self.train_n+self.test_n]])
            
        train_df = pd.DataFrame(train_data, columns=['file_path', 'label'])
        test_df = pd.DataFrame(test_data, columns=['file_path', 'label'])        

        train_df.to_csv(f'./data/train{self.train_n}.csv', index=False)
        test_df.to_csv(f'./data/test{self.test_n}.csv', index=False)

class ImbalancedData:

    def __init__(self, root, train_n=300, test_n=100, seed=0, transform=None):
        self.transform = transform
        self.root = Path(root)
        self.train_n = train_n
        self.test_n = test_n
        self.seed = seed

        train_path = f'./data/train_imbal.csv'
        test_path = f'./data/test_imbal.csv'

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            self.data_split()

        self.train_data = MiniImageNet(self.load_csv(train_path), transform=self.transform)
        self.test_data = MiniImageNet(self.load_csv(test_path), transform=self.transform)

    def load_csv(self, path):     
        f = open(path, 'r')
        lines = f.readlines()
        data = [(str(file), int(label)) for file, label in [line.strip().split(',') for line in lines[1:]]]
        return data 
    
    def data_split(self):
        random.seed(self.seed)
        train_data = []
        test_data = []
        for idx, folder in enumerate(self.root.iterdir()):
            subfiles = list(folder.iterdir())
            sel_num_train = self.train_n - idx*3
            sel_num_test = self.test_n - idx*1

            random.shuffle(subfiles)
            train_data.extend([(str(file), idx) for file in subfiles[:sel_num_train]])
            test_data.extend([(str(file), idx) for file in subfiles[sel_num_train:sel_num_train+sel_num_test]])

        train_df = pd.DataFrame(train_data, columns=['file_path', 'label'])
        test_df = pd.DataFrame(test_data, columns=['file_path', 'label'])        

        train_df.to_csv(f'./data/train_imbal.csv', index=False)
        test_df.to_csv(f'./data/test_imbal.csv', index=False)

class MiniImageNet(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn, label = self.data[idx]
        im = self.executor.submit(self.load_image, fn).result()
        label = int(label)
        return im, label

    def load_image(self, fn):
        im = cv2.imread(fn)
        im1 = cv2.resize(im, (64, 64))
        # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        if self.transform:
            im1 = self.transform(im1)
        return im1



if __name__ == '__main__':
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224))])  # ResNet 的輸入尺寸
    t = time.time()                                    
    dataset = Data('./Mini', 500, 100, transform=transform)
    train_data = dataset.train_data
    print(train_data)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=20)
    for im, label in train_loader:
        print(im.shape, label.shape)
        break
    print('Time:', time.time() - t)
