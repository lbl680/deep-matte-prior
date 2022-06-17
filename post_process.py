#!/usr/bin/env python


from __future__ import division

import logging

import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.sparse
import scipy.sparse.linalg
import scipy as sp
import scipy.ndimage
import time



def generate_trimap(alpha, K1, K2, train_mode):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    if train_mode:
        K = np.random.randint(K1, K2)
    else:
        K = np.round((K1 + K2) / 2).astype('int')

    fg = cv2.erode(fg, kernel, iterations=K)
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv2.dilate(unknown, kernel, iterations=2 * K)
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)



def format_second(secs):
    h = int(secs / 3600)
    m = int((secs % 3600) / 60)
    s = int(secs % 60)
    ss = "Exa(h:m:s):{:0>2}:{:0>2}:{:0>2}".format(h, m, s)
    return ss



def fba_fusion(alpha, img, F, B):
    F = ((alpha * img + (1 - alpha ** 2) * F - alpha * (1 - alpha) * B))
    # B = ((1 - alpha) * img + (2 * alpha - alpha**2) * B - alpha * (1 - alpha) * F)

    F = np.clip(F, 0, 1)
    # B = np.clip(B, 0, 1)
    la = 0.1

    alpha = (alpha * la + np.sum((img - B) * (F - B), 2, keepdims=True)) / (
            np.sum((F - B) * (F - B), 2, keepdims=True) + la)
    alpha = np.clip(alpha, 0, 1)
    return alpha, F, B


def fba_fusion1(alpha, img, F_0, B_0, F, B):
    sigma_fg = 10
    sigma_alpha = 2
    sigma_c = 1

    F = F_0 + 10 * alpha * (img - alpha * F - (1 - alpha) * B)
    # B = ((1 - alpha) * img + (2 * alpha - alpha**2) * B - alpha * (1 - alpha) * F)

    F = np.clip(F, 0, 1)
    # B = np.clip(B, 0, 1)

    la = sigma_c / sigma_alpha
    alpha = (alpha * la + np.sum((img - B) * (F - B), 2, keepdims=True)) / (
            np.sum((F - B) * (F - B), 2, keepdims=True) + la)
    alpha = np.clip(alpha, 0, 1)
    return alpha, F, B


def my_fba_fusion(alpha, img, F, B, iter):
    sigma_fg = 1  # 10 - min(iter, 9)
    sigma_alpha = 10  # 1 + min(iter, 9)
    sigma_c = 1

    F = alpha * sigma_fg / (sigma_c + alpha ** 2 * sigma_fg) * img + sigma_c / (sigma_c + alpha ** 2 * sigma_fg) * F - (
                sigma_fg * alpha * (1 - alpha)) / (sigma_c + alpha ** 2 * sigma_fg) * B
    # B = ((1 - alpha) * img + (2 * alpha - alpha**2) * B - alpha * (1 - alpha) * F)

    F = np.clip(F, 0, 1)
    # B = np.clip(B, 0, 1)

    # la = 0.1
    la = sigma_c / sigma_alpha
    alpha = (alpha * la + np.sum((img - B) * (F - B), 2, keepdims=True)) / (
            np.sum((F - B) * (F - B), 2, keepdims=True) + la)
    alpha = np.clip(alpha, 0, 1)
    return alpha, F, B

import os

if __name__ == '__main__':
    # fba fusion
    img_root = 'D:/lbl/2021hanjia/real_data/img_4/' # the root of input images
    bg_root = 'D:/lbl/2021hanjia/real_data/back_4/' # the root of input bgs
    pred_alpha_root = 'D:/lbl/ddip/result/result_real4/result/' # the root of predict alphas predicted by train.py
    pred_fg_root = 'D:/lbl/ddip/result/result_real4/pred_fg/' # the root of predict fgs predicted by train.py
    store_root = 'D:/lbl/ddip/result/result_real4/' # the root that will store the results
    new_alpha_root = os.path.join(store_root, 'fusion_my')
    new_fg_root = os.path.join(store_root, 'fg_fusion_my')
    if not os.path.exists(store_root):
        os.mkdir(store_root)
    if not os.path.exists(new_alpha_root):
        os.mkdir(new_alpha_root)
    if not os.path.exists(new_fg_root):
        os.mkdir(new_fg_root)

    img_names = os.listdir(pred_alpha_root)
    iter_num = 15
    eps = 1 / 1000000.
    t0 = time.time()

    for i, img_name in enumerate(img_names):
        # if i > 600:
        img_path = os.path.join(img_root, img_name)
        bg_path = os.path.join(bg_root, img_name)
        pred_alpha_path = os.path.join(pred_alpha_root, img_name)
        pred_fg_path = os.path.join(pred_fg_root, img_name)

        img = cv2.imread(img_path) / 255.
        bg = cv2.imread(bg_path) / 255.
        pred_alpha = cv2.imread(pred_alpha_path) / 255.
        if len(pred_alpha.shape) == 2:
            pred_alpha = pred_alpha[:, :, np.newaxis]
        elif pred_alpha.shape[2] == 3:
            pred_alpha = pred_alpha[:, :, 0:1]
        assert len(pred_alpha.shape) == 3
        pred_fg = cv2.imread(pred_fg_path) / 255.


        print(pred_alpha.shape)
        tmp = np.concatenate([pred_alpha, pred_alpha, pred_alpha], axis=2)
        pred_fg[tmp > 0.99] = img[tmp > 0.99]
        new_alpha, new_fg, _ = my_fba_fusion(pred_alpha, img, pred_fg, bg, 0)
        # new_alpha, new_fg, _ = fba_fusion(pred_alpha, img, pred_fg, bg)
        for j in range(iter_num - 1):
            # new_alpha = guided_filter(img, new_alpha, r, eps)
            new_alpha[new_alpha > 0.99] = 1
            new_alpha[new_alpha < 0.01] = 0
            new_alpha, new_fg, _ = my_fba_fusion(new_alpha, img, new_fg, bg, j + 1)
            # new_alpha, new_fg, _ = fba_fusion(new_alpha, img, new_fg, bg)
            # new_alpha[new_alpha>0.99] = 1
            # new_alpha[new_alpha<0.01] = 0
        new_alpha[new_alpha > 0.99] = 1
        new_alpha[new_alpha < 0.01] = 0
        new_alpha_path = os.path.join(new_alpha_root, img_name)
        new_fg_path = os.path.join(new_fg_root, img_name)
        cv2.imwrite(new_alpha_path, (new_alpha * 255).astype(np.uint8))
        cv2.imwrite(new_fg_path, (new_fg * 255).astype(np.uint8))

        t = time.time() - t0
        print('{} time: {}'.format(img_name, format_second(t)))



