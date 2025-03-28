import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import *
from dataset import *
from train import *

from minN import MinN

def runTest():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    network = SiameseNetwork()
    network.load_state_dict(torch.load("./model/latest_trainedmodel_fruit.pt"))
    network.eval()
    network.to(device)

    trans = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),  # 标准化
    ])
    train_data = []

    for label in tqdm(os.listdir('fruit30_split/train')):
        for img_name in os.listdir(f'fruit30_split/train/{label}'):
            img_path = f'fruit30_split/train/{label}/{img_name}'
            img = Image.open(img_path)
            img = trans(img)
            train_data.append((img, label))

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

    #Give each label a standard output
    trainedDataset = SiameseDataset(img_label_list = train_data, forTrain = False)
    trainedDataloader = DataLoader(trainedDataset, batch_size=1, shuffle=False)

    
    labelSet = []
    label_cnt_dict = {}

    M = 8
    N = 10

    for i, data in tqdm(enumerate(trainedDataloader, 0)):
        sample, label = data
        if label_cnt_dict.get(label, 0) >=M:
            continue
        if i%1000 == 0:
            print("Go through"+ str(i) +  "images")
        sample = sample.to(device)
        if label not in label_cnt_dict:
            labelSet.append((network(sample).data, label))
            label_cnt_dict[label] = 1
        elif label_cnt_dict[label] < M:
            labelSet.append((network(sample).data, label))
            label_cnt_dict[label] += 1
    print('Standard output setting: Done.')

    test_data = []

    for label in tqdm(os.listdir('fruit30_split/val')):
        for img_name in os.listdir(f'fruit30_split/val/{label}'):
            img_path = f'fruit30_split/val/{label}/{img_name}'
            img = Image.open(img_path)
            img = trans(img)
            test_data.append((img, label))

    # for super_class in tqdm(os.listdir('cifar100/test')):
    #     if ".DS_Store" in super_class: continue
    #     for sub_class in os.listdir(f'cifar100/test/{super_class}'):
    #         if ".DS_Store" in sub_class: continue
    #         for img_name in os.listdir(f'cifar100/test/{super_class}/{sub_class}'):
    #             img_path = f'cifar100/test/{super_class}/{sub_class}/{img_name}'
    #             img = Image.open(img_path)
    #             img = trans(img)
    #             label = super_class + sub_class
    #             test_data.append((img, label))

    #Go through all trainning data and test the accuracy
    testDataset = SiameseDataset(img_label_list = test_data, forTrain = False)
    testDataloader = DataLoader(testDataset, batch_size=1, shuffle=False)

    total_count = 0
    correct_count = 0
    for i, data in enumerate(testDataloader, 0):
        sample, label = data
        sample = sample.to(device)
        output = network(sample).data
        Minn = MinN(N)
        for vec, v_label in labelSet:
            currDis = (output - vec).pow(2).sum(1)
            Minn.add(currDis, v_label)
        
        curr_label = Minn.get()
        total_count += 1 
        if label == curr_label:
            correct_count += 1
        if total_count % 100 == 0:
            print('The total correctness rate = %.1f%%' %(correct_count/total_count*100))

    correct_count -= len(labelSet)
    total_count -= len(labelSet)
    print('The total correctness rate = %.1f%%' %(correct_count/total_count*100))

if __name__ == '__main__':
    runTest()