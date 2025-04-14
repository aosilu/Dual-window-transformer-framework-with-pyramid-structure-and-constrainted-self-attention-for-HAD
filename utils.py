import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
#from s_vit import model_vit
import time
import scipy.io as sio
from PIL import Image
import cv2
import matplotlib.pyplot as plt
def train_padding(img, out_win, r, c, bands):
    radius = int((out_win - 1) / 2)
    out_win = radius
    img_padding = np.zeros((r + 2*out_win, c + 2*out_win, bands))
    img_padding[out_win:r+out_win, out_win:c+out_win, :]=img
    img_padding[0:out_win, out_win:c+out_win, :] = np.flip(img[0:out_win, :, :], 1)    #上边镜像
    img_padding[r+out_win:r+2*out_win, out_win:c+out_win, :] = np.flip(img[r-out_win:r, :, :], 1)#下边镜像
    img_padding[out_win:r+out_win, 0:out_win, :] = np.flip(img[:, 0:out_win, :], 0) #左镜像
    img_padding[out_win:r+out_win, c+out_win:c+2*out_win, :] = np.flip(img[:, c-out_win:c, :], 0)  # 右镜像
    img_padding[0:out_win, 0:out_win, :] = np.flip(img[0:out_win, 0:out_win, :],  (0, 1))  #左上
    img_padding[0:out_win, c+out_win:c+2*out_win, :] = np.flip(img[0:out_win, c-out_win:c, :], (0, 1))#右上
    img_padding[r+out_win:r+2*out_win, 0:out_win, :] = np.flip(img[r-out_win:r, 0:out_win, :], (0, 1))#左下
    img_padding[r+out_win:r+2*out_win, c+out_win:c+2*out_win, :] = np.flip(img[r-out_win:r, c-out_win:c, :], (0, 1))#右下
    return img_padding

def bbox_mask_generator(inner, outer):
    mask = np.ones([outer]*2)
    zero_index = int((outer-inner)/2)
    mask[zero_index:-zero_index, zero_index:-zero_index] = 0
    #mask[int((outer+1)/2), int((outer+1)/2)] = 1
    mask = mask.astype(bool)
    return mask


def get_dataset(img, out_win, in_win, r, c, bands):
    img = img.astype(np.float32)
    data_new = []
    mask = bbox_mask_generator(in_win, out_win)
    radius = int((out_win - 1) / 2)
    #img = np.pad(img, ((radius, radius), (radius, radius)), 'reflect')
    img_padding = train_padding(img, out_win, r, c, bands)
    H = img_padding.shape[0]
    W = img_padding.shape[1]
    num_around = out_win**2 - in_win**2
    for i in range(radius, H-radius):
        for j in range(radius, W-radius):
            pixel = img_padding[i, j, None]
            bbox = img_padding[i-radius:i+radius+1, j-radius:j+radius+1]
            bbox = bbox[mask]
            #bbox = bbox[bbox[:, 0] > 0]
            #bbox_new = np.insert(bbox, 0, pixel, axis=0)
            #bbox_new = np.abs(bbox - pixel)
            bbox_new = bbox - pixel
            bbox_new = bbox_new.astype(np.float32)
            data_new.append(bbox_new)
    return data_new


def binarization(img, r, c, th, low, hi):
    for i in range(0, r):
        for j in range(0, c):
            if img[i, j] >= th:
                img[i, j] = hi
            else:
                img[i, j] = low
    return img

def lowresiez(img, th):
    r = np.size(img, 0)
    c = np.size(img, 1)
    bands = np.size(img, 2)
    img_new = np.empty((int(r*th), int(c*th), bands))
    for i in range(0, bands):
        cc = img[:, :, i]
        kk = cv2.resize(cc, dsize=None, fx=th, fy=th, interpolation=cv2.INTER_NEAREST)
        img_new[:, :, i] = kk
    return img_new

def upresize(img, th):
    r = np.size(img, 0)
    c = np.size(img, 1)
    img_new = cv2.resize(img, dsize=None, fx=th, fy=th, interpolation=cv2.INTER_NEAREST)
    return img_new


def norm_data(input):
    max_input = np.max(input)
    min_input = np.min(input)
    output = (input - min_input)/(max_input - min_input)
    return output


def z_score_normalize(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev
    #normalized_data[normalized_data > 1] = 1
    #normalized_data[normalized_data < 0] = 0
    return normalized_data


def weight_patch(weight, out_win, in_win):
    radius = int((out_win-1)/2)
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
            weight_new.append(bbox_z)
    weight_new = np.array(weight_new)
    a = np.min(weight_new)
    b = np.max(weight_new)
    #weight_new = norm_data(weight_new)
    a = 1
    return weight_new

def reduce(img):
    r = np.size(img, 0)
    c = np.size(img, 1)
    max_kk = r * c
    img = img.reshape(-1)
    kk = sorted(img, reverse=True)
    th = kk[int(0.03 * max_kk)]
    img = img.reshape(r, c)
    img[img < th] = 0
    return img


def binarization_result(img):

    r = np.size(img, 0)
    c = np.size(img, 1)
    max_kk = r * c
    img = img.reshape(-1)
    kk = sorted(img, reverse=True)
    th = kk[int(0.01 * max_kk)]
    img_r = np.ones((r, c))
    img = img.reshape(r, c)
    for i in range(r):
        for j in range(c):
            if img[i, j] < th:
                img_r[i, j] = 0
            else:
                img_r[i, j] = 1
    return img_r

def merge_stretegy(img):
    r = np.size(img, 0)
    c = np.size(img, 1)
    max_kk = r * c
    img = img.reshape(-1)
    kk = sorted(img, reverse=True)
    th = kk[int(0.008 * max_kk)]
    img_r = np.zeros((r, c))
    img = img.reshape(r, c)
    for i in range(r):
        for j in range(c):
            if img[i, j] > th:
                img_r[i, j] = 1
            else:
                img_r[i, j] = img[i, j]
    return img_r

def remove_isolated_points_stretegy(imag, min_size):
    # 定义孤立点的最小尺寸
    # 假设 image 是一个二维NumPy数组，其中0表示黑色，1表示白色
    # image = np.array([...])
    image = binarization_result(imag)
    #plt.figure()
    #plt.imshow(image, cmap='jet')
    #plt.show()
    image_8bit = cv2.convertScaleAbs(image)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_8bit)

    # 移除孤立点
    for i in range(1, num_labels):
        x, y, w, h, _ = stats[i]
        area = w * h
        if area < min_size:
            image[labels == i] = 0
    return image