import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from vit import model_vit
import time
import scipy.io as sio
from scipy.io import savemat
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from utils import get_dataset, lowresiez, upresize, norm_data, z_score_normalize, weight_patch, reduce, binarization_result, remove_isolated_points_stretegy, merge_stretegy
from utils import binarization
from get_weight import gweight
import torch.nn.functional as F
import cv2


def plot_roc_curve(y, pred, title, pos_label_num):
    # ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    fpr, tpr, threshold = roc_curve(y, pred, pos_label=pos_label_num)
    roc_auc = auc(fpr, tpr)
    print('roc_auc:', roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate ')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    plt.savefig('result/'+title+'.svg', dpi=600)
    plt.show()

def main(test_in, weightp):
    r = np.size(test_in, 0)
    c = np.size(test_in, 1)
    bands = np.size(test_in, 2)
    weightp = torch.FloatTensor(weightp)

    test = get_dataset(test_in, out_win, in_win, r, c, bands)
    test = np.array(test)
    test = torch.FloatTensor(test)


    weight_test=Data.TensorDataset(test, weightp)
    test_loader = Data.DataLoader(weight_test, batch_size=batch_size,shuffle=False)

    start = time.time()
    detection_results = []
    with torch.no_grad():
        for batch_data_f, weight in test_loader:
            batch_data_f = batch_data_f.cuda()
            weight = weight.cuda()
            preds = model(batch_data_f, weight)
            detection_results = np.append(detection_results, preds.data.cpu().numpy())
    end = time.time()
    ttt = end - start
    print('test time:', ttt)
    detection_results = norm_data(detection_results)
    output = detection_results.reshape((r, c))
    max_kk = r * c
    output = output.reshape(-1)
    kk = sorted(output, reverse=True)
    th = kk[int(0.01 * max_kk)]
    output = output.reshape(r, c)
    output[output < th] = 0

    #savemat(results_path, {results_name: output})
    '''
    plt.figure()
    plt.imshow(output, cmap='jet')
    plt.show()
    '''
    return output


batch_size = 3000
out_win = 11
in_win = 9
th = 0.4
hi = 1
low = 1e-2
num = 4
# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model_vit.to(device)
model.load_state_dict(torch.load('model.ckpt', map_location=device))
model.eval()

file_path = 'data/'
data_name = 'dataset'   #micai1  smallplane22
data_path = file_path + data_name + '.mat'
gt_name = 'dataset'   #micai1_gt    gt_smallplane22  junglecamouflage
gt_path = file_path + gt_name + '.mat'
#results_file_path = 'result/'
#results_name = data_name
#results_path = results_file_path + results_name + '.mat'
test_in = sio.loadmat(data_path)['a']
test_label_in = sio.loadmat(gt_path)['a_gt'] #junglecamouflage4_gt
test_in = test_in.astype(np.float32)
test_label_in = test_label_in.astype(np.float32)
#micai1
test_in = test_in[0:496,0:296, :]
test_label_in = test_label_in[0:496,0:296]
#jc2  4
#test_in = test_in[0:1000, 0:1000, :]
#test_label_in = test_label_in[0:1000, 0:1000]
layer_t = []
layer_s = []

ro = np.size(test_in, 0)
co = np.size(test_in, 1)


for i in range(num-1, -1, -1):
    if i == num-1:
        test_in_new = lowresiez(test_in, 0.5 ** i)
        train_label_in_new = cv2.resize(test_label_in, dsize=None, fx=0.5 ** i, fy=0.5 ** i, interpolation=cv2.INTER_NEAREST)
        weightp, res_sae = gweight(test_in_new, out_win, in_win)
        res_sae_up = upresize(res_sae, 2**i)
        res_sae_up = cv2.blur(res_sae_up, ksize=(11, 11))
        #res_sae_up = reduce(res_sae_up)
        #res_sae_up = remove_isolated_points_stretegy(res_sae_up, 4000)
        #res_sae_up = merge_stretegy(res_sae_up)
        result = main(test_in_new, weightp)
        kk = reduce(result)
        kk = upresize(kk, 2**i)
        #kk = reduce(kk)
        kk = cv2.blur(kk, ksize=(11, 11))
        #kk = reduce(kk)
        #kk = remove_isolated_points_stretegy(kk, 2000)
        #kk = merge_stretegy(kk)
        layer_t.append(kk)
        layer_s.append(res_sae_up)
        result = upresize(result, 2)
    else:
        test_in_new = lowresiez(test_in, 0.5 ** i)
        train_label_in_new = cv2.resize(test_label_in, dsize=None, fx=0.5 ** i, fy=0.5 ** i, interpolation=cv2.INTER_NEAREST)
        _, res_sae = gweight(test_in_new, out_win, in_win)
        res_sae_up = upresize(res_sae, 2 ** i)
        res_sae_up = cv2.blur(res_sae_up, ksize=(11, 11))
        #res_sae_up = reduce(res_sae_up)
        #res_sae_up = remove_isolated_points_stretegy(res_sae_up, 4000)
        #res_sae_up = merge_stretegy(res_sae_up)
        layer_s.append(res_sae_up)
        result = reduce(result)
        result_weightp = norm_data(result+res_sae)
        result_weightp_patch = weight_patch(result_weightp, out_win, in_win)
        result = main(test_in_new,  result_weightp_patch)
        kk = reduce(result)
        kk = upresize(kk, 2**i)
        #kk = reduce(kk)
        kk = cv2.blur(kk, ksize=(11, 11))
        #kk = reduce(kk)
        #kk = remove_isolated_points_stretegy(kk, 2000)
        #kk = merge_stretegy(kk)
        layer_t.append(kk)
        result = upresize(result, 2)


layer_t = np.array(layer_t)
layer_s = np.array(layer_s)
layer = np.ones((num, ro, co))


for i in range(0, num):
    layer[i] = layer_t[i]+layer_s[i]
    layer[i] = norm_data(layer[i])

plt.figure()
for i in range(0, num):
    plt.subplot(3, num, i+1)
    plt.imshow(layer_t[i, :, :], cmap='jet')
    plt.subplot(3, num, i+num+1)
    plt.imshow(layer_s[i, :, :], cmap='jet')
    plt.subplot(3, num, i+2*num+1)
    plt.imshow(layer[i, :, :], cmap='jet')
plt.show()

detection_results = np.ones((ro, co))


for i in range(num-1, num):
    detection_results = detection_results+layer[i]
detection_results = norm_data(detection_results)

#detection_results = remove_isolated_points_stretegy(detection_results, 200)

#detection_results = z_score_normalize(detection_results)

#detection_results[detection_results >= 1] = 1
#detection_results[detection_results <= 0] = 0

plt.figure()
plt.imshow(detection_results, cmap='jet')
plt.show()

sio.savemat('./after_result/0_train_result_a.mat', {'result_a': detection_results})
test_label_in = test_label_in.reshape(-1)
detection_results = detection_results.reshape(-1)
plot_roc_curve(test_label_in, detection_results, data_name, 1)

