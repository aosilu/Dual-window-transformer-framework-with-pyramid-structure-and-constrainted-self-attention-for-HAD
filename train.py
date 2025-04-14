import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from vit import model_vit
import time
import scipy.io as sio
from utils import get_dataset
import matplotlib.pyplot as plt
from utils import binarization
from get_weight import gweight
from PIL import Image
from utils import get_dataset, lowresiez, upresize, norm_data, z_score_normalize, weight_patch, reduce
import cv2



print(torch.version)

def plot_loss(arry_train,arry_test):
    line1, = plt.plot(range(0,len(arry_train)),arry_train,'r.-')
    #line2, = plt.plot(range(0,len(arry_test)),arry_test,'b.-')
    plt_title = 'BATCH_SIZE = 1500;EPOCH = 200'
    plt.title(plt_title)
    #plt.legend(handles=[line1, line2], labels=["train_loss", "test_loss"], loc="upper right", fontsize=7)
    plt.ylabel('LOSS')
    plt.show()

def train_epoch(model, train_loader, criterion, optimizer, arry_train):
    detection_results = []
    for batch_idx, (batch_data, batch_target, weightp) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        weightp = weightp.cuda()

        optimizer.zero_grad()
        batch_pred = model(batch_data, weightp)
        detection_results = np.append(detection_results, batch_pred.data.cpu().numpy())
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        '''
        if (batch_idx + 1) % 100 == 0:
            print('Loss: {:.8f}'
                .format(loss.item()))
        '''
    arry_train.append(loss.item())
    detection_results = norm_data(detection_results)
    return detection_results


def main(train_data_in, train_label_in, weightp):
    r = np.size(train_data_in, 0)
    c = np.size(train_data_in, 1)
    bands = np.size(train_data_in, 2)
    '''
    weight = binarization(train_label_in, r, c, th, low, hi)
    train_in_weight = np.zeros((r, c, bands))
    for i in range (0, bands):
        train_in_weight[:, :, i] = train_data_in[:, :, i] * weight
    '''
    x_train = get_dataset(train_data_in, out_win, in_win, r, c, bands)
    x_train = np.array(x_train)
    x_train = torch.FloatTensor(x_train)  # 400000 24 89

    train_label_in = train_label_in.reshape(-1)
    x_train_label = torch.FloatTensor(train_label_in)

    weightp = torch.FloatTensor(weightp)

    Label_train = Data.TensorDataset(x_train, x_train_label, weightp)
    label_train_loader = Data.DataLoader(Label_train, batch_size=batch_size, shuffle=True)


    # train
    arry_train = []
    arry_test = []

    print("start training")
    tic_train = time.time()
    for epoch in range(epoches):
        # scheduler.step()
        # train model
        model.train()
        print('Epoch [{}/{}]'
              .format(epoch + 1, epoches, ))
        output = train_epoch(model, label_train_loader, criterion, optimizer, arry_train)
        # scheduler.step()
    #toc_train = time.time()
    #print("Training ends   Running Time: {:.2f}".format(toc_train - tic_train))
    # arry_train = np.array(arry_train)
    #plot_loss(arry_train, arry_test)

    output = output.reshape((r, c))
    return output


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameter
epoches = 1
learning_rate = 5e-4
weight_decay = 0
gamma = 0.9
batch_size = 3000
out_win = 11
in_win = 9

num = 4

# input data
file_path = 'data/'
data_name = 'micai2'
data_path = file_path+data_name+'.mat'
gt_name = 'micai2_gt'
gt_path = file_path+gt_name+'.mat'
train_data_in = sio.loadmat(data_path)['micai2']
train_label_in = sio.loadmat(gt_path)['micai2_gt']
train_data_in = train_data_in.astype(np.float32)
train_label_in = train_label_in.astype(np.float32)  #400 1000  89

#train_data_in = train_data_in[0:744,:, :]
#train_label_in = train_label_in[0:744,:]

# create model

model = model_vit.to(device)

# criterion
criterion = nn.BCELoss().cuda()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoches//10, gamma=gamma)

for i in range(num-1, -1, -1):
    if i == num-1:
        train_in_new = lowresiez(train_data_in, 0.5 ** i)
        train_label_in_new = cv2.resize(train_label_in, dsize=None, fx=0.5 ** i, fy=0.5 ** i, interpolation=cv2.INTER_NEAREST)
        weightp,_ = gweight(train_in_new, out_win, in_win)
        result = main(train_in_new, train_label_in_new, weightp)
        result = upresize(result, 2)
    else:
        train_in_new = lowresiez(train_data_in, 0.5 ** i)
        train_label_in_new = cv2.resize(train_label_in, dsize=None, fx=0.5 ** i, fy=0.5 ** i, interpolation=cv2.INTER_NEAREST)
        _, res_sae = gweight(train_in_new, out_win, in_win)
        #result = reduce(result)
        result_weightp = norm_data(result+res_sae)
        result_weightp_patch = weight_patch(result_weightp, out_win, in_win)
        result = main(train_in_new, train_label_in_new, result_weightp_patch)
        result = upresize(result, 2)
torch.save(model.state_dict(), 'model.ckpt')

