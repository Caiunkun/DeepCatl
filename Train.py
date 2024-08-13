import os
import xlwt
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import math
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import random_split
from Dataset.script import SSDataset_690
from fightingcv_attention.attention.ECAAttention import ECAAttention

class Constructor:

    def __init__(self, model, model_name='d_ssca'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5, verbose=1)
        self.loss_function = nn.BCELoss()

        self.batch_size = 64
        self.epochs = 14

    def learn(self, TrainLoader, ValidateLoader):

        path = os.path.abspath(os.curdir)
        for epoch in range(self.epochs):
            self.model.train()
            ProgressBar = tqdm(TrainLoader)
            for data in ProgressBar:
                self.optimizer.zero_grad()
                ProgressBar.set_description("Epoch %d" % epoch)
                seq, shape, label = data
                output = self.model(seq.unsqueeze(1).to(self.device), shape.unsqueeze(1).to(self.device))   #训练
                loss = self.loss_function(output, label.float().to(self.device))
                ProgressBar.set_postfix(loss=loss.item())
                loss.backward()
                self.optimizer.step()
            valid_loss = []
            self.model.eval()
            with torch.no_grad():
                for valid_seq, valid_shape, valid_labels in ValidateLoader:
                    valid_output = self.model(valid_seq.unsqueeze(1).to(self.device), valid_shape.unsqueeze(1).to(self.device))
                    valid_labels = valid_labels.float().to(self.device)
                    valid_loss.append(self.loss_function(valid_output, valid_labels).item())
                valid_loss_avg = torch.mean(torch.Tensor(valid_loss))
                self.scheduler.step(valid_loss_avg)
        torch.save(self.model.state_dict(), path + '\\' + self.model_name + '.pth')
        print('\n---Finish Learn---\n')

    def inference(self, TestLoader):

        path = os.path.abspath(os.curdir)
        self.model.load_state_dict(torch.load(path + '\\' + self.model_name + '.pth', map_location='cpu'))

        predicted_value = []
        ground_label = []


        # model.eval()的作用是不启用 Batch Normalization 和 Dropout。
        self.model.eval()
        # 遍历测试数据加载器，每次迭代获取一个测试样本的序列数据、形状信息和标签。
        for seq, shape, label in TestLoader:
            output = self.model( seq.unsqueeze(1).to(self.device), shape.unsqueeze(1).to(self.device) )
            # """ To scalar"""
            # 看不懂
            predicted_value.append(output.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy())
            ground_label.append(label.squeeze(dim=0).squeeze(dim=0).detach().numpy())

        print('\n---Finish Inference---\n')
        return predicted_value, ground_label

    def measure(self, predicted_value, ground_label):
        accuracy = accuracy_score(y_pred=np.array(predicted_value).round(), y_true=ground_label)
        roc_auc = roc_auc_score(y_score=predicted_value, y_true=ground_label)

        precision, recall, _ = precision_recall_curve(probas_pred=predicted_value, y_true=ground_label)
        pr_auc = auc(recall, precision)

        print('\n---Finish Measure---\n')

        return accuracy, roc_auc, pr_auc

    def run(self, samples_file_name, ratio=0.8):

        Train_Validate_Set = SSDataset_690(samples_file_name, False)          #加载数据  返回数据集对象用于训练和验证

        """divide Train samples and Validate samples"""                       #划分数据集
        Train_Set, Validate_Set = random_split(dataset=Train_Validate_Set,
                                               lengths=[math.ceil(len(Train_Validate_Set) * ratio),
                                                        len(Train_Validate_Set) -
                                                        math.ceil(len(Train_Validate_Set) * ratio)],
                                               generator=torch.Generator().manual_seed(0))
        # 一、🌱📚 DataLoader的参数说明 📚🌱
        # dataset(必需): 用于加载数据的数据集，通常是torch.utils.data.Dataset的子类实例。
        # batch_size(可选): 每个批次的数据样本数。默认值为1。             当一个Epoch的样本（也就是所有的训练样本）数量可能太过庞大（对于计算机而言），就需要把它分成多个小块，也就是就是分成多个Batch 来进行训练。
        # shuffle(可选): 是否在每个周期开始时打乱数据。默认为False。
        # sampler(可选): 定义从数据集中抽取样本的策略。如果指定，则忽略shuffle参数。
        # batch_sampler(可选): 与sampler类似，但一次返回一个批次的索引。不能与batch_size、shuffle和sampler同时使用。
        # num_workers(可选): 用于数据加载的子进程数量。默认为0，意味着数据将在主进程中加载。
        # collate_fn(可选): 如何将多个数据样本整合成一个批次。通常不需要指定。
        # drop_last(可选): 如果数据集大小不能被批次大小整除，是否丢弃最后一个不完整的批次。默认为False。

        #   训练集
        TrainLoader = loader.DataLoader(dataset=Train_Set, drop_last=True,
                                        batch_size=self.batch_size, shuffle=True, num_workers=0)
        #   测试集
        ValidateLoader = loader.DataLoader(dataset=Validate_Set, drop_last=True,
                                           batch_size=self.batch_size, shuffle=False, num_workers=0)
        #   验证集
        TestLoader = loader.DataLoader(dataset=SSDataset_690(samples_file_name, True),
                                       batch_size=1, shuffle=False, num_workers=0)


        #   开始训练
        self.learn(TrainLoader, ValidateLoader)


        predicted_value, ground_label = self.inference(TestLoader)

        accuracy, roc_auc, pr_auc = self.measure(predicted_value, ground_label)

        print('\n---Finish Run---\n')
        print('acc,roc,prauc',accuracy,roc_auc,pr_auc)

        return accuracy, roc_auc, pr_auc


from models.DanQ import DanQ
from models.D_SSCA import d_ssca
from models.CRPTS import crpts  
from models.DeepCatl import deepcatl


Train = Constructor(model=deepcatl())
model=deepcatl()
strings = ["wgEncodeAwgTfbsBroadDnd41CtcfUniPk"]





listacc = []
listauc = []
listprauc = []


for i in range(len(strings)):
    print("'"+strings[i]+"'")
    lista,listb,listc = Train.run(samples_file_name=strings[i])

    listacc.append(lista)
    listauc.append(listb)
    listprauc.append(listc)


f = xlwt.Workbook('encoding = utf-8')  # 设置工作簿编码
sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)  # 创建sheet工作表

for i in range(len(listacc)):
    sheet1.write(i, 0, listacc[i])  # 写入数据参数对应 行, 列, 值

for i in range(len(listauc)):
    sheet1.write(i, 1, listauc[i])  # 写入数据参数对应 行, 列, 值

for i in range(len(listprauc)):
    sheet1.write(i, 2, listprauc[i])  # 写入数据参数对应 行, 列, 值
f.save('text.xls')  # 保存.xls到当前工作目录



