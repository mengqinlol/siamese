from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random

class SiameseDataset(Dataset):
    def __init__(self, img_label_list, forTrain, transform = None):
        """
            image_label_list: 一个列表，每个元素为 (image, label)
            transform: 数据预处理和增强
        """
        self.image_label_list = img_label_list
        self.transform = transform
        self.forTrain = forTrain
        if forTrain:
            self.label_to_img = {}
            for img, label in self.image_label_list:
                self.label_to_img.setdefault(label, []).append(img)
            print(len(self.label_to_img.keys()))
        
        self.label_to_label_possibilities =[[1.0 for _ in range(len(self.label_to_img.keys()))] for _ in range(len(self.label_to_img.keys()))]

    def set_zero_loss_label_combinations(self, label1_idx, label2_idx):
        if self.label_to_label_possibilities[label1_idx][label2_idx] < 0.2:
            return
        self.label_to_label_possibilities[label1_idx][label2_idx] *= 0.9
        self.label_to_label_possibilities[label2_idx][label1_idx] *= 0.9

    def __len__(self):
        return len(self.image_label_list)
    def __getitem__(self, idx):
        if self.forTrain:
            anchor_item = self.image_label_list[idx]
            anchor = anchor_item[0]
            ancher_label = anchor_item[1]
            ancher_label_idx = list(self.label_to_img.keys()).index(ancher_label)
            positive = random.choice(self.label_to_img[ancher_label])
            # while positive == anchor:
            #     positive = random.choice(self.label_to_img[ancher_label])
            
            while True:
                negative_label_idx = random.randint(0, len(self.label_to_img.keys())-1)
                negative_label = list(self.label_to_img.keys())[negative_label_idx]
                if negative_label == ancher_label: continue
                if random.random() < self.label_to_label_possibilities[ancher_label_idx][negative_label_idx]: break
                # print("possibilities" + str(self.label_to_label_possibilities[ancher_label_idx][negative_label_idx]))

            negative = random.choice(self.label_to_img[negative_label])
            if self.transform:
                anchor = self.transform(anchor)
                positive = self.transform(positive)
                negative = self.transform(negative)
            return anchor, positive, negative, ancher_label_idx, negative_label_idx
        else:
            img = self.image_label_list[idx][0]
            label = self.image_label_list[idx][1]
            if self.transform:
                img = self.transform(img)
            return img, label

class ToTensor(object):
    def __call__(self,sample):
        image = sample
        image = image.transpose((2,0,1))
        return image


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, list)
        self.output_size = output_size

    def __call__(self, image):
        return transform.resize(image, (self.output_size[0], self.output_size[1]))
