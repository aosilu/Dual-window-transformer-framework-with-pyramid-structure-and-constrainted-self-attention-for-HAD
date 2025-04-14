import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn, optim
import scipy.io as sio
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import type_of_target
import time
import cv2
from utils import bbox_mask_generator
from utils import norm_data, z_score_normalize, reduce, binarization_result, remove_isolated_points_stretegy
def gweight(X_train_in, out_win, in_win):
    # 自编码器
    class AutoEncoder(nn.Module):
        def __init__(self):
            super(AutoEncoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(89, 50),
                nn.LeakyReLU(),
                nn.Linear(50, 30),
                nn.LeakyReLU(),

            )
            self.decoder = nn.Sequential(
                nn.Linear(30, 50),
                nn.LeakyReLU(),
                nn.Linear(50, 89),
                nn.Tanh()
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded
    def weight_patch(weight, outwin, inwin):
        radius = int((outwin-1)/2)
        reflect = cv2.copyMakeBorder(weight, radius, radius, radius, radius, cv2.BORDER_REFLECT)
        mask = bbox_mask_generator(in_win, out_win)
        weight_new = []
        H = reflect.shape[0]
        W = reflect.shape[1]
        num_around = out_win ** 2 - in_win ** 2
        for i in range(radius, H - radius):
            for j in range(radius, W - radius):
                pixel = reflect[i, j]
                bbox = reflect[i - radius:i + radius + 1, j - radius:j + radius + 1]
                bbox = bbox[mask]
                # bbox = bbox[bbox[:, 0] > 0]
                # bbox_new = np.insert(bbox, 0, pixel, axis=0)
                # bbox_new = np.abs(bbox - pixel)
                bbox_new = bbox - pixel
                bbox_f = (bbox_new)*pixel
                bbox_f = bbox_f.reshape(-1, 1)
                bbox_z = np.dot(bbox_f, bbox_f.T)
                bbox_z = bbox_z.astype(np.float32)
                weight_new.append(bbox_z)
        weight_new = np.array(weight_new)
        a = np.min(weight_new)
        b = np.max(weight_new)
        #weight_new = norm_data(weight_new)
        a = 1
        return weight_new


    def plot_roc_curve(y, pred, title, pos_label_num):
        # ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
        fpr, tpr, threshold = roc_curve(y, pred, pos_label=pos_label_num)
        roc_auc = auc(fpr, tpr)
        end = time.time()
        tttime = end - start
        print('roc_auc:', roc_auc)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.4f)  time = %0.4f' % (roc_auc, tttime))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig('result/' + title + '.svg', dpi=600)
        plt.show()


    # 参数设置
    epochs = 1
    # 读取数据
    start = time.time()
    X_train_in = X_train_in.astype(float)
    X_train_in = X_train_in / np.max(X_train_in)
    r = np.size(X_train_in, 0)
    c = np.size(X_train_in, 1)
    bands = np.size(X_train_in, 2)
    X_train_input = X_train_in.reshape(-1, bands)
    # 数据类型转换
    trainData = torch.FloatTensor(X_train_input)

    # 构建张量数据集
    train_dataset = TensorDataset(trainData, trainData)

    trainDataLoader = DataLoader(dataset=train_dataset, batch_size=2000, shuffle=False)

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 网络放到GPU上
    autoencoder = AutoEncoder().to(device)
    # 初始化
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()
    loss_train = np.zeros((epochs, 1))

    # 训练
    for epoch in range(epochs):
        # 不需要label，所以用一个占位符"_"代替
        for batchidx, (x, _) in enumerate(trainDataLoader):
            x = x.to(device)
            # 编码和解码
            encoded, decoded = autoencoder(x)
            # 计算loss
            loss = loss_func(decoded, x)
            # 更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_train[epoch, 0] = loss.item()
        print('epoch=%d finished' % (epoch + 1))
        '''
        print('%d  index=%d  Epoch: %04d, Training loss=%.8f' %
    
              (p, index, epoch + 1, loss.item()))
        '''

    '''
    # 绘制loss曲线
    fig = plt.figure(figsize=(6, 3))
    ax = plt.subplot(1, 1, 1)
    ax.grid()
    ax.plot(loss_train, color=[245 / 255, 124 / 255, 0 / 255], linestyle='-', linewidth=2)
    ax.set_xlabel('Epoches')
    ax.set_ylabel('Loss')
    plt.show()
    '''
    # 利用训练好的自编码器重构测试数据
    trainData = trainData.to(device)
    _, decodedTestdata = autoencoder(trainData)
    decodedTestdata = decodedTestdata.double()
    reconstructedData = decodedTestdata.detach().cpu().numpy()
    output = reconstructedData.reshape(r, c, bands)

    error = np.abs(X_train_in - output)
    # sio.savemat('result/smallplane11.mat', {'smallplane11':r})

    result_sorce = np.linalg.norm(error, ord=None, axis=2)
    result_sorce = norm_data(result_sorce)

    #result_sorce = z_score_normalize(result_sorce)

    #sio.savemat('result/result_smallplane11_l2.mat', {'result_smallplane11_l2': result_sorce})

    #plt.figure()
    #plt.imshow(result_sorce, cmap='jet')
    #plt.show()

    #result_sorce = binarization_result(result_sorce)
    #result_sorce = cv2.blur(result_sorce, ksize=(5, 5))

    #plt.figure()
    #plt.imshow(result_sorce, cmap='jet')
    #plt.show()


    weightp = weight_patch(result_sorce, out_win, in_win)


    return weightp, result_sorce

