import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from model import *
from dataset import *
import sys
from tqdm import tqdm
from PIL import Image


#Configuration
IMAGE_SIZE = [220, 220]
BATCH_SIZE = 32
MODELNAME = "CNN_model_imagenet"+".pt"
EPOCH = 300


class TripletLoss(nn.Module):
    def __init__(self, margin = 1):
        super(TripletLoss,self).__init__()
        self.margin= margin

    def forward(self, anchor, positive, negative):
        APDistance = (anchor - positive).pow(2).sum(1)
        ANDistance = (anchor - negative).pow(2).sum(1)

        # print(APDistance, ANDistance)
        loss = F.relu(APDistance - ANDistance + self.margin)
        return loss.sum()
    
def save_loss(loss):
    with open('res_loss.txt', 'a') as f:
        f.write(str(loss) + '\n')

def runTrain():

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    
    net = torch.load("latest_CNN_trainedmodel_animal copy.pt")
    net = net.to(device)

    print(device)

    trans = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Resize(IMAGE_SIZE),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
    ])
    train_data = []

    # 读取训练数据
    # for super_class in tqdm(os.listdir('cifar100/train')):
    #     if ".DS_Store" in super_class: continue
    #     for sub_class in os.listdir(f'cifar100/train/{super_class}'):
    #         if ".DS_Store" in sub_class: continue
    #         for img_name in os.listdir(f'cifar100/train/{super_class}/{sub_class}'):
    #             img_path = f'cifar100/train/{super_class}/{sub_class}/{img_name}'
    #             img = Image.open(img_path)
    #             img = trans(img)
    #             label = super_class + sub_class
    #             train_data.append((img, label))

    # for label in tqdm(os.listdir('fruit30_split/train')):
    #     for img_name in os.listdir(f'fruit30_split/train/{label}'):
    #         img_path = f'fruit30_split/train/{label}/{img_name}'
    #         img = Image.open(img_path)
    #         img = trans(img)
    #         train_data.append((img, label))

    dataset_path = 'datasets/imagenet-tiny/train'

    for label in tqdm(os.listdir(dataset_path)):
        cnt = 0
        for img_name in os.listdir(f'{dataset_path}/{label}'):
            img_path = f'{dataset_path}/{label}/{img_name}'
            img = Image.open(img_path).convert('RGB')
            img = trans(img)
            train_data.append((img, label))
            cnt += 1

    trainDataset = SiameseDataset(img_label_list = train_data, forTrain = True)
    trainDataloader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = TripletLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.000001)
    for epoch in range(EPOCH):
        loss_sum = 0.0
        loss_to_show = 0.0
        pbar = tqdm(enumerate(trainDataloader), total=len(trainDataloader), desc=f'Epoch {epoch+1}/{EPOCH}')
        for i, data in pbar:
            anchor, positive, negative = data
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            anchor_out, positive_out, negative_out = net.forward_triple(anchor, positive, negative)
            optimizer.zero_grad()
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            loss_to_show = loss.item()
            pbar.set_postfix(loss=f'{loss_to_show:.10f}', refresh=True) 
        print(f'Epoch {epoch+1}/{EPOCH} loss: {loss_sum / len(trainDataloader) :.10f}')
        save_loss(loss_sum / len(trainDataloader))
        if epoch % 10 == 0:
            torch.save(net.state_dict(), "./model/"+f'_{epoch+1}'+MODELNAME+'h')
            print('Checkpoint saved.')
            torch.save(net, "./model/latest_"+MODELNAME)
    print('Training done.')
    torch.save(net.state_dict(), "./model/"+MODELNAME)
