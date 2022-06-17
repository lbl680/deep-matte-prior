import cv2
import os
import numpy as np
from loss_functions import CovarianceLayer, VarianceLayer, MeanLayer, YIQGNGCLoss, GradientLoss
from colorsys import rgb_to_yiq
from functions import np_to_torch, torch_to_np, pil_to_np, np_to_pil
import torch.nn.functional as F
import torch
from scipy.stats import multivariate_normal as mn

# if __name__ == '__main__':
#     # generate trimap using corr
#     img_root = 'D:/lbl/myBackgroundMattingData/merged_test20_20_512/img/'
#     bg_root = 'D:/lbl/myBackgroundMattingData/merged_test20_20_512/back/'
#     alpha_root = 'D:/lbl/myBackgroundMattingData/merged_test20_20_512/alpha/'
#     res_root = 'D:/lbl/myBackgroundMattingData/result/result/init/initialization_tri_bigpatch'
#     # res_fg_root = 'D:/lbl/myBackgroundMattingData/result/result/fg_initialization'
#
#     if not os.path.exists(res_root):
#         os.mkdir(res_root)
#     # if not os.path.exists(res_fg_root):
#     #     os.mkdir(res_fg_root)
#
#     # res_root1 = os.path.join(res_root, 'res')
#     # # res_root_alfg = os.path.join(res_root, 'alfg')
#     # if not os.path.exists(res_root1):
#     #     os.mkdir(res_root1)
#     # if not os.path.exists(res_root_alfg):
#     #     os.mkdir(res_root_alfg)
#
#     covar = CovarianceLayer().cuda()
#     comp_var = VarianceLayer(channels=3).cuda()
#     comp_GNGC = YIQGNGCLoss(shape=5).cuda()
#     comp_mean = MeanLayer(channels=3).cuda()
#
#     img_names = os.listdir(bg_root)
#
#     # comp ori tri
#     for img_name in img_names:
#         print(img_name)
#         img_path = os.path.join(img_root, img_name)
#         bg_path = os.path.join(bg_root, img_name)
#         alpha_path = os.path.join(alpha_root, img_name)
#         img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
#         bg = cv2.imread(bg_path, cv2.COLOR_BGR2RGB)
#         alpha = cv2.imread(alpha_path)[:, :, 0:1]
#
#         # comp ori seg
#         dis = np.sqrt(np.mean((img.astype(np.float32) - bg.astype(np.float32)) ** 2, axis=2))
#         thres = np.min(dis)  # + 0.1 * (np.max(dis) - np.min(dis))
#         erzhi = np.zeros_like(dis)
#         erzhi[dis > thres] = 1
#
#         # preprocess(dilate)
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         ero = cv2.erode(erzhi, kernel, iterations=10)
#         trimap = ero + (erzhi - ero) * 0.5
#
#         # comp corr
#         x = pil_to_np(img)
#         y = pil_to_np(bg)
#         x = np_to_torch(x).cuda()
#         y = np_to_torch(y).cuda()
#         # print(x.shape)
#         # print(y.shape)
#         # x_g = rgb_to_yiq(x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :])[0]  # take the Y part
#         # y_g = rgb_to_yiq(y[:, 0:1, :, :], y[:, 1:2, :, :], y[:, 2:3, :, :])[0]  # take the Y part
#         # x_g = x[:, 0:1, :, :]
#         # y_g = y[:, 0:1, :, :]
#         #
#         # cov = covar(x_g.cuda(), y_g.cuda()) ** 2
#         # cov = torch_to_np(cov)[0]
#         # cv2.normalize(cov, cov, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#         # cov = cov.astype(np.uint8)
#
#         # dis = torch.sqrt(torch.mean((x - y) ** 2, dim=1, keepdim=True))
#         # mask = (dis > 0.001).type(torch.cuda.FloatTensor)
#         mask = torch.ones_like(x[:, 0:1, :, :])
#         # print(mask.shape)
#
#         covs = []
#         for i in range(3):
#             x_g = x[:, i:i + 1, :, :]  # take the Y part
#             y_g = y[:, i:i + 1, :, :]  # take the Y part
#             cov, _, _ = comp_GNGC(x_g * 2 - 1, y_g * 2 - 1, mask)
#             covs.append(cov)
#         # print('gngc.shape: {}'.format(cov.shape))
#         cov = covs[0]
#         for i in range(1, 3):
#             cov += covs[1]
#         cov /= 3
#         cov = torch_to_np(cov)[0]
#         res_root2 = os.path.join(res_root, 'res')
#         if not os.path.exists(res_root2):
#             os.mkdir(res_root2)
#         res_path = os.path.join(res_root2, img_name)
#         tmp = np.zeros_like(cov)
#         tmp[cov > 0] = cov[cov > 0]
#         cv2.imwrite(res_path, np.clip(tmp * 255, 0, 255).astype(np.uint8))
#
#         print(np.min(cov))
#         print(np.max(cov))
#         # res_thres = (cov > np.min(cov) + 0.9 * (np.max(cov) - np.min(cov))).astype(np.float32)
#         res_thres = (cov > 0.9 * np.max(cov)).astype(np.float32)
#
#         res_root2 = os.path.join(res_root, 'res_thres')
#         if not os.path.exists(res_root2):
#             os.mkdir(res_root2)
#         res_path = os.path.join(res_root2, img_name)
#         cv2.imwrite(res_path, (res_thres * 255).astype(np.uint8))
#         # res_thres = res_thres.astype(np.uint8) * 255
#         # if np.sum(cov != cov) != 0:
#         #     print('nan is in cov')
#         # cv2.normalize(cov, cov, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#         # cov = cov.astype(np.uint8)
#
#         # refine trimap according to the corr
#         print('ero.shape: {}'.format(ero.shape))
#         print('res_thres.shape: {}'.format(res_thres.shape))
#         pad_num = (ero.shape[0] - res_thres.shape[0]) // 2
#         print(pad_num)
#         tmp = np.zeros_like(ero)
#         print('tmp.shape: {}'.format(tmp.shape))
#         tmp[pad_num:ero.shape[0] - pad_num, pad_num:ero.shape[1] - pad_num] = res_thres
#         area = tmp * ero
#         unknown = cv2.dilate(area, kernel, iterations=15)
#         trimap = trimap * (1 - unknown) + 0.5 * unknown
#         trimap = (trimap * 255).astype(np.uint8)
#
#         res_root2 = os.path.join(res_root, 'tri')
#         if not os.path.exists(res_root2):
#             os.mkdir(res_root2)
#         res_path = os.path.join(res_root2, img_name)
#         cv2.imwrite(res_path, trimap)


# for img_name in img_names:
#     print(img_name)
#     img_path = os.path.join(img_root, img_name)
#     bg_path = os.path.join(bg_root, img_name)
#     alpha_path = os.path.join(alpha_root, img_name)
#     img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
#     bg = cv2.imread(bg_path, cv2.COLOR_BGR2RGB)
#     alpha = cv2.imread(alpha_path)[:, :, 0:1]
#     # fg = img - bg
#     # print('img.shape: {}'.format(img.shape))
#     # print('bg.shape: {}'.format(bg.shape))
#
#     # # compute covarance and visualize
#     x = pil_to_np(img)
#     y = pil_to_np(bg)
#     alpha = pil_to_np(alpha)
#     x = np_to_torch(x).cuda()
#     y = np_to_torch(y).cuda()
#     # print(x.shape)
#     # print(y.shape)
#     x_g = rgb_to_yiq(x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :])[0]  # take the Y part
#     y_g = rgb_to_yiq(y[:, 0:1, :, :], y[:, 1:2, :, :], y[:, 2:3, :, :])[0]  # take the Y part
#     # x_g = x[:, 0:1, :, :]
#     # y_g = y[:, 0:1, :, :]
#     #
#     # cov = covar(x_g.cuda(), y_g.cuda()) ** 2
#     # cov = torch_to_np(cov)[0]
#     # cv2.normalize(cov, cov, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     # cov = cov.astype(np.uint8)
#
#     # dis = torch.sqrt(torch.mean((x - y) ** 2, dim=1, keepdim=True))
#     # mask = (dis > 0.001).type(torch.cuda.FloatTensor)
#     mask = torch.ones_like(x[:, 0:1, :, :])
#     # print(mask.shape)
#
#     cov, vv, c = comp_GNGC(x_g, y_g, mask)
#     # print('gngc.shape: {}'.format(cov.shape))
#     cov = torch_to_np(cov)[0]
#     print(np.min(cov))
#     print(np.max(cov))
#     res_thres = (cov > (np.min(cov) + 0.9 * (np.max(cov) - np.min(cov)))).astype(np.float32)
#     res_thres = res_thres.astype(np.uint8) * 255
#     if np.sum(cov != cov) != 0:
#         print('nan is in cov')
#     cv2.normalize(cov, cov, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     cov = cov.astype(np.uint8)
#
#     vv = torch_to_np(vv)[0]
#     if np.sum(vv != vv) != 0:
#         print('nan is in vv')
#     print(np.min(vv))
#     print(np.max(vv))
#     # cv2.normalize(vv, vv, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     vv_show = np.zeros_like(vv)
#     vv_show[vv < 0.000001] = 255
#     # vv = vv.astype(np.uint8)
#     vv_show = vv_show.astype(np.uint8)
#
#     c = torch_to_np(c)[0]
#     if np.sum(c != c) != 0:
#         print('nan is in c')
#
#     # print(cov.shape)
#     # print(np.min(cov))
#     # print(np.max(cov))
#
#     res_path = os.path.join(res_root1, img_name)
#     cv2.imwrite(res_path, cov)
#
#     res_root2 = os.path.join(res_root, 'res_thres')
#     if not os.path.exists(res_root2):
#         os.mkdir(res_root2)
#     res_path = os.path.join(res_root2, img_name)
#     cv2.imwrite(res_path, res_thres)
#
#     # x = torch_to_np(x)
#     # print(x.shape)
#     # x = (x.transpose(1, 2, 0) * 255).astype(np.uint8)
#     # # x = np_to_pil(x)
#     # print(x.shape)
#     # res_path = os.path.join(res_root_alfg, img_name)
#     # cv2.imwrite(res_path, x)
#
#
#     # # add two feature(mean and variance)
#     # mean_img = comp_mean(x.detach())
#     # mean_bg = comp_mean(y.detach())
#     # pad_num = (x.shape[2] - mean_img.shape[2]) // 2
#     # mean_img = F.pad(mean_img, [pad_num, pad_num, pad_num, pad_num])
#     # print('mean_img.shape: {}'.format(mean_img.shape))
#     # print('x.shape: {}'.format(x.shape))
#     # assert mean_img.shape == x.shape
#     # mean_bg = F.pad(mean_bg, [pad_num, pad_num, pad_num, pad_num])
#     # assert mean_bg.shape == y.shape
#     # mean_img = torch_to_np(mean_img)
#     # print('mean_img.shape: {}'.format(mean_img.shape))
#     # cv2.normalize(mean_img, mean_img, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     # mean_bg = torch_to_np(mean_bg)
#     # cv2.normalize(mean_bg, mean_bg, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     #
#     # var_img = comp_var(x.detach())
#     # var_bg = comp_var(y.detach())
#     # pad_num = (x.shape[2] - var_img.shape[2]) // 2
#     # var_img = F.pad(var_img, [pad_num, pad_num, pad_num, pad_num])
#     # print('var_img.shape: {}'.format(var_img.shape))
#     # assert var_img.shape == x.shape
#     # var_bg = F.pad(var_bg, [pad_num, pad_num, pad_num, pad_num])
#     # assert var_bg.shape == y.shape
#     # var_img = torch_to_np(var_img)
#     # print('var_img.shape: {}'.format(var_img.shape))
#     # cv2.normalize(var_img, var_img, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     # var_bg = torch_to_np(var_bg)
#     # cv2.normalize(var_bg, var_bg, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     #
#     # print('img.shape: {}'.format(img.shape))
#     # img = np.concatenate([img.astype(np.float32), mean_img.transpose(1, 2, 0), var_img.transpose(1, 2, 0)], axis=2)
#     # bg = np.concatenate([bg.astype(np.float32), mean_bg.transpose(1, 2, 0), var_bg.transpose(1, 2, 0)], axis=2)
#     # # img = np.concatenate([img.astype(np.float32), mean_img.transpose(1, 2, 0)], axis=2)
#     # # bg = np.concatenate([bg.astype(np.float32), mean_bg.transpose(1, 2, 0)], axis=2)
#     # print('img.shape: {}'.format(img.shape))
#
#
#
#     # # normal
#     # dis = np.sqrt(np.mean((img.astype(np.float32) - bg.astype(np.float32)) ** 2, axis=2))
#     # # dis = np.sqrt((img[:, :, 4] - bg[:, :, 4]) ** 2)
#     # print(dis.shape)
#     # dis = dis * 255 / np.max(dis)
#     # dis = dis.astype(np.uint8)
#     # _, dis_ostu = cv2.threshold(dis[:, :, np.newaxis], 0, 255, cv2.THRESH_OTSU)
#     # print('dis_ostu.shape: {}'.format(dis_ostu.shape))
#     # print('dis_ostu.min: {}, max: {}'.format(np.min(dis_ostu), np.max(dis_ostu)))
#     # print(dis.shape)
#     # print(np.min(dis))
#     # print(np.max(dis))
#     # # thres = np.min(dis) #+ 0.1 * (np.max(dis) - np.min(dis))
#     # # erzhi = np.zeros_like(dis)
#     # # erzhi[dis > thres] = 1
#     # # erzhi = erzhi * 255
#     # # erzhi = erzhi.astype(np.uint8)
#     # # print(np.min(dis))
#     # # print(np.max(dis))
#     #
#     # # tri
#     # trimap = np.ones_like(dis) * 255
#     # trimap[dis_ostu.astype(np.float32) < 0.001] = 128
#     # trimap[dis < 0.001] = 0
#     #
#     # dis = dis * 255 / np.max(dis)
#     # dis = dis.astype(np.uint8)
#     # # print(np.min(dis))
#     # # print(np.max(dis))
#     # # # _, init = cv2.threshold(dis, 0, np.max(dis), cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     # # init = (dis > 0).astype(np.uint8) * 255
#     # # print(init.shape)
#     # # print(np.max(init))
#     #
#     # res_path = os.path.join(res_root, img_name)
#     # res_fg_path = os.path.join(res_fg_root, img_name)
#     # # cv2.imwrite(res_path, init)
#     # # cv2.imwrite(res_path, erzhi)
#     # # cv2.imwrite(res_fg_path, fg)
#     # cv2.imwrite(res_path, trimap)


# kl-divergence
import numpy as np
from scipy.linalg import logm
from torch import nn
import time


def format_second(secs):
    h = int(secs / 3600)
    m = int((secs % 3600) / 60)
    s = int(secs % 60)
    ss = "Exa(h:m:s):{:0>2}:{:0>2}:{:0>2}".format(h, m, s)
    return ss

def multivar_continue_KL_divergence(p_u, p_sig, q_u, q_sig):
    # u:c*1 sig:c*c

    # eps = 0.0000001eps * np.identity(q_sig.shape[0]
    q_sig_inv = np.linalg.inv(q_sig)
    a = np.log(np.abs(np.linalg.det(q_sig) / np.linalg.det(p_sig)))
    b = np.trace(np.dot(q_sig_inv, p_sig))
    c = np.dot(np.dot(np.transpose(q_u - p_u), q_sig_inv), (q_u - p_u))
    n = p_sig.shape[0]
    kl = 0.5 * (a - n + b + c)
    # if np.isnan(kl):
    #     print(np.linalg.det(q_sig))
    #     print(np.linalg.det(p_sig))
    #     print(p_u)
    #     print(p_sig)
    #     print(q_u)
    #     print(q_sig)
    #     print('q_sig_inv: {}'.format(q_sig_inv))
    #     print('a: {:.5f}'.format(a))
    #     print('b: {:.5f}'.format(b))
    #     print('c: {:.5f}'.format(c))
    #     print('kl: {:.5f}'.format(kl))
    # if 0 < kl < 10:
    #     print(p_u)
    #     print(p_sig)
    #     print(q_u)
    #     print(q_sig)
    #     print('q_sig_inv: {}'.format(q_sig_inv))
    #     print('a: {:.5f}'.format(a))
    #     print('b: {:.5f}'.format(b))
    #     print('c: {:.5f}'.format(c))
    #     print('kl: {:.5f}'.format(kl))
    return kl


def js_loss(mean1, mean2, var1, var2, n_samples=10 ** 5):
    ''' JS-divergence between two multinomial Gaussian distribution
    mean1    (list): List of mean vectors [M1_1, M2_1, ...]
    mean2    (list): List of mean vectors [M1_2, M2_2, ...]
    log_var1 (list): List of log_var vectors [LV1_1, LV2_1, ...]
    log_var2 (list): List of log_var vectors [LV2_2, LV2_2, ...]
    returns the list of Jensen-Shannon divergences [JS1, JS2, ...]
    '''


    p = mn(mean=mean1, cov=var1)
    q = mn(mean=mean2, cov=var2)
    div = js(p, q, n_samples)

    return div

def js(p, q, n_samples=10 ** 5):
    ''' Jensen-Shannon divergence with Monte-Carlo approximation
    p          (scipy.stats): Statistical continues function
    q          (scipy.stats): Statistical continues function
    n_samples:         (int): Number of samples for Monte-Carlo approximation
    returns divergence [0., 1.] between the multinomial continues distributions p and q
    '''

    # Sample from p and q
    X, Y = p.rvs(size=n_samples, random_state=0), q.rvs(size=n_samples, random_state=0)
    # Evaluate p and q at samples from p
    p_X, q_X = p.pdf(X), q.pdf(X)
    # Evaluate p and q at samples from q
    p_Y, q_Y = p.pdf(Y), q.pdf(Y)
    # Evaluate the mixtures at samples from p and q
    log_mix_X, log_mix_Y = np.log2(p_X + q_X), np.log2(p_Y + q_Y)

    # calculate the Jensen-Shannon entropy
    JS = (np.log2(p_X).mean() - (log_mix_X.mean() - np.log2(2))
            + np.log2(q_Y).mean() - (log_mix_Y.mean() - np.log2(2))) / 2
    return JS

def main():
    # generate trimap using corr
    img_root = 'D:/lbl/2021hanjia/real_data/img_4/'
    bg_root = 'D:/lbl/2021hanjia/real_data/back_4/'
    # alpha_root = 'D:/lbl/myBackgroundMattingData/merged_test20_20_512/alpha/'
    res_root = 'D:/lbl/2021hanjia/real_data/initialization_4/'
    # res_fg_root = 'D:/lbl/myBackgroundMattingData/result/result/fg_initialization'

    if not os.path.exists(res_root):
        os.mkdir(res_root)
    # if not os.path.exists(res_fg_root):
    #     os.mkdir(res_fg_root)

    # res_root1 = os.path.join(res_root, 'res')
    # # res_root_alfg = os.path.join(res_root, 'alfg')
    # if not os.path.exists(res_root1):
    #     os.mkdir(res_root1)
    # if not os.path.exists(res_root_alfg):
    #     os.mkdir(res_root_alfg)

    patch_size = 5
    eps = 0.000001
    comp_covar = CovarianceLayer(patch_size=patch_size).cuda()
    comp_var = VarianceLayer().cuda()
    comp_mean = MeanLayer(patch_size=patch_size).cuda()
    comp_grad = GradientLoss().cuda()

    img_names = os.listdir(bg_root)

    # comp ori tri
    for img_name in img_names[160:]:
        t0 = time.time()
        print(img_name)
        img_path = os.path.join(img_root, img_name)
        bg_path = os.path.join(bg_root, img_name)
        # alpha_path = os.path.join(alpha_root, img_name)
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        bg = cv2.imread(bg_path, cv2.COLOR_BGR2RGB)
        # alpha = cv2.imread(alpha_path)[:, :, 0:1]

        # comp ori seg
        dis = np.sqrt(np.mean((img.astype(np.float32) - bg.astype(np.float32)) ** 2, axis=2))
        thres = np.min(dis)  # + 0.1 * (np.max(dis) - np.min(dis))
        erzhi = np.zeros_like(dis)
        erzhi[dis > thres] = 1

        # preprocess(dilate)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ero = cv2.erode(erzhi, kernel, iterations=10)
        trimap = ero + (erzhi - ero) * 0.5

        # comp corr
        x = pil_to_np(img)
        y = pil_to_np(bg)
        x = np_to_torch(x).cuda()
        y = np_to_torch(y).cuda()
        # print(x.shape)
        # print(y.shape)
        print('x.min: {:.5f}'.format(torch.min(x)))
        print('x.max: {:.5f}'.format(torch.max(x)))

        # comp first grad
        grad_x1, grad_y1 = comp_grad(x, 1)
        grad_x2, grad_y2 = comp_grad(y, 1)
        print('gradx1.min: {:.5f}'.format(torch.min(grad_x1)))
        print('gradx1.max: {:.5f}'.format(torch.max(grad_x1)))

        # comp second grad
        grad_x1_2, grad_y1_2 = comp_grad(x, 2)
        grad_x2_2, grad_y2_2 = comp_grad(y, 2)
        # print('gradx1.min: {:.5f}'.format(torch.min(grad_x1_2)))
        # print('gradx1.max: {:.5f}'.format(torch.max(grad_x1_2)))

        x = torch.cat([x, grad_x1, grad_y1, grad_x1_2, grad_y1_2], dim=1)
        y = torch.cat([y, grad_x2, grad_y2, grad_x2_2, grad_y2_2], dim=1)

        c = x.shape[1]

        x_u = []
        for i in range(c):
            x_u.append(comp_mean(x[:, i:i+1, :, :]))
        x_u = torch.cat(x_u, dim=1)
        print('x_u.shape: {}'.format(x_u.shape))
        print('x_u.min: {:.5f}'.format(torch.min(x_u)))
        print('x_u.max: {:.5f}'.format(torch.max(x_u)))

        y_u = []
        for i in range(c):
            y_u.append(comp_mean(y[:, i:i+1, :, :]))
        y_u = torch.cat(y_u, dim=1)
        print('y_u.shape: {}'.format(y_u.shape))
        print('y_u.min: {:.5f}'.format(torch.min(y_u)))
        print('y_u.max: {:.5f}'.format(torch.max(y_u)))

        x_sig = torch.zeros((c, c, x.shape[2]-patch_size//2*4, x.shape[3]-patch_size//2*4))
        # x_sig = torch.zeros((c, c, x_u.shape[2], x_u.shape[3]))
        for i in range(c):
            for j in range(c):
                x_sig[i, j] = comp_covar(x[:, i:i+1, :, :], x[:, j:j+1, :, :])[0, 0, :, :]
                if i == j:
                    x_sig[i, j] += eps
        # x_sig = torch.cat(x_sig, dim=1)
        print('x_sig.shape: {}'.format(x_sig.shape))
        print('x_sig.min: {:.5f}'.format(torch.min(x_sig)))
        print('x_sig.max: {:.5f}'.format(torch.max(x_sig)))
        print('x_sig.isnan: {}'.format(True in np.isnan(x_sig)))


        y_sig = torch.zeros((c, c, y.shape[2]-patch_size//2*4, y.shape[3]-patch_size//2*4))
        # y_sig = torch.zeros((c, c, y_u.shape[2], y_u.shape[3]))
        for i in range(c):
            for j in range(c):
                y_sig[i, j] = comp_covar(y[:, i:i+1, :, :], y[:, j:j+1, :, :])[0, 0, :, :]
                if i == j:
                    y_sig[i, j] += eps
        # x_sig = torch.cat(x_sig, dim=1)
        print('y_sig.shape: {}'.format(y_sig.shape))
        print('y_sig.min: {:.5f}'.format(torch.min(y_sig)))
        print('y_sig.max: {:.5f}'.format(torch.max(y_sig)))
        print('y_sig.isnan: {}'.format(True in np.isnan(y_sig)))

        # y_sig = []
        # for i in range(3):
        #     for j in range(3):
        #         y_sig.append(comp_covar(y[:, i:i+1, :, :], y[:, j:j+1, :, :]))
        # y_sig = torch.cat(y_sig, dim=1)
        # print('y_sig.shape: {}'.format(y_sig.shape))

        pad_num = (x_u.shape[2] - x_sig.shape[2]) // 2
        print('pad_num: {}'.format(pad_num))
        x_u = x_u[:, :, pad_num:x_u.shape[2]-pad_num, pad_num:x_u.shape[3]-pad_num]
        y_u = y_u[:, :, pad_num:y_u.shape[2]-pad_num, pad_num:y_u.shape[3]-pad_num]
        print('x_u.shape: {}'.format(x_u.shape))
        print('y_u.shape: {}'.format(y_u.shape))

        # torch to np
        x_u = torch_to_np(x_u)
        y_u = torch_to_np(y_u)
        x_sig = x_sig.detach().cpu().numpy()
        y_sig = y_sig.detach().cpu().numpy()
        print('x_u.shape: {}'.format(x_u.shape))
        print('y_u.shape: {}'.format(y_u.shape))
        print('x_sig.shape: {}'.format(x_sig.shape))
        print('y_sig.shape: {}'.format(y_sig.shape))

        # reshape
        # x_u = np.transpose(x_u, (1, 2, 0))
        # y_u = np.transpose(y_u, (1, 2, 0))
        print('u_shape: {}'.format(x_u[:, 0, 0].shape))
        print('sig_shape: {}'.format(x_sig[:, :, 0, 0].shape))

        # comp kl
        kl = np.zeros((x_u.shape[1], x_u.shape[2], 1))
        print('kl_shape: {}'.format(kl.shape))
        for i in range(x_u.shape[1]):
            t1 = time.time()
            for j in range(x_u.shape[2]):
                # print(i, j)
                kl[i, j, :] = multivar_continue_KL_divergence(x_u[:, i, j], x_sig[:, :, i, j], y_u[:, i, j], y_sig[:, :, i, j])
                kl[i, j, :] += multivar_continue_KL_divergence(y_u[:, i, j], y_sig[:, :, i, j], x_u[:, i, j], x_sig[:, :, i, j])
                # kl[i, j, :] = js_loss(x_u[:, i, j], y_u[:, i, j], x_sig[:, :, i, j], y_sig[:, :, i, j], n_samples=100)
                # print(kl[i, j, :])
                # print(format_second(time.time()-t1))
                # kl[i, j, :] = 1 / (kl[i, j, :] + 0.1)
                # kl[i, j, :] = 1 / (kl[i, j, :] + 50)
                # kl[i, j, :] = kl[i, j, :] - 1
                # kl[i, j, :] = 1 - kl[i, j, :]
                # kl[i, j, :] = 0.5 * multivar_continue_KL_divergence(x_u[:, i, j], x_sig[:, :, i, j], (y_u[:, i, j]+x_u[:, i, j])/2, (y_sig[:, :, i, j]+x_sig[:, :, i, j])/4)
                # kl[i, j, :] += 0.5 * multivar_continue_KL_divergence(y_u[:, i, j], y_sig[:, :, i, j], (x_u[:, i, j]+y_u[:, i, j])/2, (x_sig[:, :, i, j]+y_sig[:, :, i, j])/4)

        print('kl.min: {:.5f}'.format(np.min(kl)))
        print('kl.max: {:.5f}'.format(np.max(kl)))

        kl = 1 / (kl + 100)
        # kl = 1 - kl

        print('kl.min: {:.5f}'.format(np.min(kl)))
        print('kl.max: {:.5f}'.format(np.max(kl)))

        cov = (kl[:, :, 0] - np.min(kl)) / (np.max(kl) - np.min(kl))
        print('cov.min: {:.5f}'.format(np.min(cov)))
        print('cov.max: {:.5f}'.format(np.max(cov)))
        res_root2 = os.path.join(res_root, 'res')
        if not os.path.exists(res_root2):
            os.mkdir(res_root2)
        res_path = os.path.join(res_root2, img_name)
        cv2.imwrite(res_path, np.clip(cov * 255, 0, 255).astype(np.uint8))

        print(np.min(cov))
        print(np.max(cov))
        # res_thres = (cov > np.min(cov) + 0.9 * (np.max(cov) - np.min(cov))).astype(np.float32)
        cov = kl[:, :, 0]
        res_thres = (cov > 0.5 * np.max(cov)).astype(np.float32)

        res_root2 = os.path.join(res_root, 'res_thres')
        if not os.path.exists(res_root2):
            os.mkdir(res_root2)
        res_path = os.path.join(res_root2, img_name)
        cv2.imwrite(res_path, (res_thres * 255).astype(np.uint8))
        # res_thres = res_thres.astype(np.uint8) * 255
        # if np.sum(cov != cov) != 0:
        #     print('nan is in cov')
        # cv2.normalize(cov, cov, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # cov = cov.astype(np.uint8)

        # refine trimap according to the corr
        print('ero.shape: {}'.format(ero.shape))
        print('res_thres.shape: {}'.format(res_thres.shape))
        pad_num = (ero.shape[0] - res_thres.shape[0]) // 2
        print(pad_num)
        tmp = np.zeros_like(ero)
        print('tmp.shape: {}'.format(tmp.shape))
        tmp[pad_num:ero.shape[0] - pad_num, pad_num:ero.shape[1] - pad_num] = res_thres
        area = tmp * ero
        unknown = cv2.dilate(area, kernel, iterations=15)
        trimap = trimap * (1 - unknown) + 0.5 * unknown
        trimap = (trimap * 255).astype(np.uint8)

        res_root2 = os.path.join(res_root, 'tri')
        if not os.path.exists(res_root2):
            os.mkdir(res_root2)
        res_path = os.path.join(res_root2, img_name)
        cv2.imwrite(res_path, trimap)

        t = time.time() - t0
        print(format_second(t))

def main_patchdis():
    # generate trimap using corr
    img_root = 'D:/lbl/myBackgroundMattingData/merged_test20_20_512/img/'
    bg_root = 'D:/lbl/myBackgroundMattingData/merged_test20_20_512/back/'
    alpha_root = 'D:/lbl/myBackgroundMattingData/merged_test20_20_512/alpha/'
    res_root = 'D:/lbl/myBackgroundMattingData/result/result/init/initialization_tri_gradcorr'
    # res_fg_root = 'D:/lbl/myBackgroundMattingData/result/result/fg_initialization'

    if not os.path.exists(res_root):
        os.mkdir(res_root)
    # if not os.path.exists(res_fg_root):
    #     os.mkdir(res_fg_root)

    # res_root1 = os.path.join(res_root, 'res')
    # # res_root_alfg = os.path.join(res_root, 'alfg')
    # if not os.path.exists(res_root1):
    #     os.mkdir(res_root1)
    # if not os.path.exists(res_root_alfg):
    #     os.mkdir(res_root_alfg)

    covar = CovarianceLayer().cuda()
    comp_var = VarianceLayer(channels=3).cuda()
    comp_GNGC = YIQGNGCLoss(shape=5).cuda()
    comp_mean = MeanLayer(channels=3).cuda()
    comp_grad = GradientLoss().cuda()

    img_names = os.listdir(bg_root)

    # comp ori tri
    for img_name in img_names:
        print(img_name)
        img_path = os.path.join(img_root, img_name)
        bg_path = os.path.join(bg_root, img_name)
        alpha_path = os.path.join(alpha_root, img_name)
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        bg = cv2.imread(bg_path, cv2.COLOR_BGR2RGB)
        alpha = cv2.imread(alpha_path)[:, :, 0:1]

        # comp ori seg
        dis = np.sqrt(np.mean((img.astype(np.float32) - bg.astype(np.float32)) ** 2, axis=2))
        thres = np.min(dis)  # + 0.1 * (np.max(dis) - np.min(dis))
        erzhi = np.zeros_like(dis)
        erzhi[dis > thres] = 1


        # preprocess(dilate)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ero = cv2.erode(erzhi, kernel, iterations=10)
        trimap = ero + (erzhi - ero) * 0.5

        # comp corr
        x = pil_to_np(img)
        y = pil_to_np(bg)
        x = np_to_torch(x).cuda()
        y = np_to_torch(y).cuda()
        # print(x.shape)
        # print(y.shape)
        # x_g = rgb_to_yiq(x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :])[0]  # take the Y part
        # y_g = rgb_to_yiq(y[:, 0:1, :, :], y[:, 1:2, :, :], y[:, 2:3, :, :])[0]  # take the Y part
        # x_g = x[:, 0:1, :, :]
        # y_g = y[:, 0:1, :, :]
        #
        # cov = covar(x_g.cuda(), y_g.cuda()) ** 2
        # cov = torch_to_np(cov)[0]
        # cv2.normalize(cov, cov, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # cov = cov.astype(np.uint8)

        # dis = torch.sqrt(torch.mean((x - y) ** 2, dim=1, keepdim=True))
        # mask = (dis > 0.001).type(torch.cuda.FloatTensor)
        mask = torch.ones_like(x[:, 0:1, :, :])
        # print(mask.shape)

        # comp grad dis
        grad_x1, grad_y1 = comp_grad(x)
        grad_x2, grad_y2 = comp_grad(y)
        print('gradx1.min: {:.5f}'.format(torch.min(grad_x1)))
        print('gradx1.max: {:.5f}'.format(torch.max(grad_x1)))

        # dis_gradx =
        # pad_num = (x.shape[2] - grad_x1.shape[2]) // 2
        # grad_x1_l = np.zeros_like(x)
        # grad_x1_l[:, :, pad_num:x.shape[2] - pad_num, pad_num:x.shape[3] - pad_num] = grad_x1
        # grad_y1_l = np.zeros_like(x)
        # grad_y1_l[:, :, pad_num:x.shape[2] - pad_num, pad_num:x.shape[3] - pad_num] = grad_y1
        # grad_x2_l = np.zeros_like(x)
        # grad_x2_l[:, :, pad_num:x.shape[2] - pad_num, pad_num:x.shape[3] - pad_num] = grad_x2
        # grad_y2_l = np.zeros_like(x)
        # grad_y2_l[:, :, pad_num:x.shape[2] - pad_num, pad_num:x.shape[3] - pad_num] = grad_y2

        # combine grad
        grad_1 = torch.sqrt(0.5 * grad_x1**2 + 0.5 * grad_y1**2)
        grad_2 = torch.sqrt(0.5 * grad_x2**2 + 0.5 * grad_y2**2)
        # thres small grad
        grad_1[(grad_1 > -0.1) & (grad_1 < 0.1)] = 0
        grad_2[(grad_2 > -0.1) & (grad_2 < 0.1)] = 0
        print('grad1.min: {:.5f}'.format(torch.min(grad_1)))
        print('grad2.max: {:.5f}'.format(torch.max(grad_2)))

        # covs = []
        # for i in range(3):
        #     x_g = x[:, i:i + 1, :, :]  # take the Y part
        #     y_g = y[:, i:i + 1, :, :]  # take the Y part
        #     cov, vv, c = comp_GNGC(grad_1, grad_2, mask)
        #     # cov[cov < 0] = 0
        #     covs.append(cov)
        # # print('gngc.shape: {}'.format(cov.shape))
        # cov = covs[0]
        # for i in range(1, 3):
        #     cov += covs[1]
        # cov /= 3

        cov, vv, c = comp_GNGC(grad_1, grad_2, mask)
        cov = torch_to_np(cov)[0]
        res_root2 = os.path.join(res_root, 'res')
        if not os.path.exists(res_root2):
            os.mkdir(res_root2)
        res_path = os.path.join(res_root2, img_name)
        # tmp = np.zeros_like(cov)
        # tmp[cov > 0] = cov[cov > 0]
        # cov = (cov+1)/2
        cv2.imwrite(res_path, np.clip(cov * 255, 0, 255).astype(np.uint8))

        print(np.min(cov))
        print(np.max(cov))
        # res_thres = (cov > np.min(cov) + 0.9 * (np.max(cov) - np.min(cov))).astype(np.float32)
        res_thres = (cov > 0.9 * np.max(cov)).astype(np.float32)

        res_root2 = os.path.join(res_root, 'res_thres')
        if not os.path.exists(res_root2):
            os.mkdir(res_root2)
        res_path = os.path.join(res_root2, img_name)
        cv2.imwrite(res_path, (res_thres * 255).astype(np.uint8))
        # res_thres = res_thres.astype(np.uint8) * 255
        # if np.sum(cov != cov) != 0:
        #     print('nan is in cov')
        # cv2.normalize(cov, cov, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # cov = cov.astype(np.uint8)

        # refine trimap according to the corr
        print('ero.shape: {}'.format(ero.shape))
        print('res_thres.shape: {}'.format(res_thres.shape))
        pad_num = (ero.shape[0] - res_thres.shape[0]) // 2
        print(pad_num)
        tmp = np.zeros_like(ero)
        print('tmp.shape: {}'.format(tmp.shape))
        tmp[pad_num:ero.shape[0] - pad_num, pad_num:ero.shape[1] - pad_num] = res_thres
        area = tmp * ero
        unknown = cv2.dilate(area, kernel, iterations=15)
        trimap = trimap * (1 - unknown) + 0.5 * unknown
        trimap = (trimap * 255).astype(np.uint8)

        res_root2 = os.path.join(res_root, 'tri')
        if not os.path.exists(res_root2):
            os.mkdir(res_root2)
        res_path = os.path.join(res_root2, img_name)
        cv2.imwrite(res_path, trimap)

        vv = torch_to_np(vv)[0]
        vv = (vv - np.min(vv)) / (np.max(vv) - np.min(vv))
        res_root2 = os.path.join(res_root, 'vv')
        if not os.path.exists(res_root2):
            os.mkdir(res_root2)
        res_path = os.path.join(res_root2, img_name)
        cv2.imwrite(res_path, vv*255)

        c = torch_to_np(c)[0]
        tmp = np.zeros_like(c)
        tmp[c < 0] = 1
        res_root2 = os.path.join(res_root, 'c_neg')
        if not os.path.exists(res_root2):
            os.mkdir(res_root2)
        res_path = os.path.join(res_root2, img_name)
        cv2.imwrite(res_path, tmp*255)

        # vv = torch_to_np(vv)[0]
        tmp = np.zeros_like(vv)
        tmp[vv < 0.0011] = 1
        res_root2 = os.path.join(res_root, 'vv_small')
        if not os.path.exists(res_root2):
            os.mkdir(res_root2)
        res_path = os.path.join(res_root2, img_name)
        cv2.imwrite(res_path, tmp*255)

        c = (c - np.min(c)) / (np.max(c) - np.min(c))
        res_root2 = os.path.join(res_root, 'c')
        if not os.path.exists(res_root2):
            os.mkdir(res_root2)
        res_path = os.path.join(res_root2, img_name)
        cv2.imwrite(res_path, c*255)

        grad_1 = torch_to_np(grad_1)[0]

        tmp = np.zeros_like(grad_1)
        tmp[(grad_1 > -0.05) & (grad_1 < 0.05)] = 1
        res_root2 = os.path.join(res_root, 'grad1_zero')
        if not os.path.exists(res_root2):
            os.mkdir(res_root2)
        res_path = os.path.join(res_root2, img_name)
        cv2.imwrite(res_path, tmp*255)

        grad_1 = np.abs(grad_1) * (1 - tmp)
        grad_1 = (grad_1 - np.min(grad_1)) / (np.max(grad_1) - np.min(grad_1))
        res_root2 = os.path.join(res_root, 'grad_1')
        if not os.path.exists(res_root2):
            os.mkdir(res_root2)
        res_path = os.path.join(res_root2, img_name)
        cv2.imwrite(res_path, grad_1*255)

        grad_2 = torch_to_np(grad_2)[0]

        tmp = np.zeros_like(grad_2)
        tmp[(grad_2 > -0.05) & (grad_2 < 0.05)] = 1
        res_root2 = os.path.join(res_root, 'grad2_zero')
        if not os.path.exists(res_root2):
            os.mkdir(res_root2)
        res_path = os.path.join(res_root2, img_name)
        cv2.imwrite(res_path, tmp*255)

        grad_2 = np.abs(grad_2) * (1 - tmp)
        grad_2 = (grad_2 - np.min(grad_2)) / (np.max(grad_2) - np.min(grad_2))
        res_root2 = os.path.join(res_root, 'grad_2')
        if not os.path.exists(res_root2):
            os.mkdir(res_root2)
        res_path = os.path.join(res_root2, img_name)
        cv2.imwrite(res_path, grad_2*255)

# if __name__ == '__main__':
#     main()

# if __name__ == '__main__':
#     # generate trimap using corr
#     img_root = 'D:/lbl/2021hanjia/real_data/img_4/'
#     bg_root = 'D:/lbl/2021hanjia/real_data/back_4/'
#     # alpha_root = 'D:/lbl/myBackgroundMattingData/merged_train_1/alpha/'
#     res_root = 'D:/lbl/2021hanjia/real_data/initialization_4/'#'D:/lbl/myBackgroundMattingData/result/result/init/initialization_tri1_fullreso_all/'
#     # res_fg_root = 'D:/lbl/myBackgroundMattingData/result/result/fg_initialization'
#
#     if not os.path.exists(res_root):
#         os.mkdir(res_root)
#     # if not os.path.exists(res_fg_root):
#     #     os.mkdir(res_fg_root)
#
#     # res_root1 = os.path.join(res_root, 'res')
#     # # res_root_alfg = os.path.join(res_root, 'alfg')
#     # if not os.path.exists(res_root1):
#     #     os.mkdir(res_root1)
#     # if not os.path.exists(res_root_alfg):
#     #     os.mkdir(res_root_alfg)
#
#     covar = CovarianceLayer().cuda()
#     comp_var = VarianceLayer(channels=1).cuda()
#     comp_GNGC = YIQGNGCLoss(shape=5).cuda()
#     comp_mean = MeanLayer(channels=1).cuda()
#     comp_grad = GradientLoss().cuda()
#
#     img_names = os.listdir(bg_root)
#
#     # comp ori tri
#     for img_name in img_names:
#         print(img_name)
#         img_path = os.path.join(img_root, img_name)
#         bg_path = os.path.join(bg_root, img_name)
#         # alpha_path = os.path.join(alpha_root, img_name)
#         img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
#         bg = cv2.imread(bg_path, cv2.COLOR_BGR2RGB)
#         # alpha = cv2.imread(alpha_path)[:, :, 0:1]
#
#         # comp ori seg
#         dis = np.sqrt(np.mean((img.astype(np.float32) / 255. - bg.astype(np.float32) / 255.) ** 2, axis=2))
#         thres = np.min(dis) + 0.05  # + 0.1 * (np.max(dis) - np.min(dis))
#         erzhi = np.zeros_like(dis)
#         erzhi[dis > thres] = 1
#
#         tmp = (dis > (np.min(dis) + 0.02 * (np.max(dis) - np.min(dis)))).astype(np.float32)
#         # # aver
#         # tmp = np_to_torch(dis).unsqueeze(0).cuda()
#         # tmp = comp_mean(tmp)
#         # tmp = torch_to_np(tmp)[0]
#         # tmp = (tmp > (np.min(tmp) + 0.05)).astype(np.float32)
#
#         res_root2 = os.path.join(res_root, 'dis')
#         if not os.path.exists(res_root2):
#             os.mkdir(res_root2)
#         res_path = os.path.join(res_root2, img_name)
#         cv2.imwrite(res_path, (dis * 255).astype(np.uint8))
#
#         res_root2 = os.path.join(res_root, 'dis_thres')
#         if not os.path.exists(res_root2):
#             os.mkdir(res_root2)
#         res_path = os.path.join(res_root2, img_name)
#         cv2.imwrite(res_path, (tmp * 255).astype(np.uint8))
#
#         # preprocess(dilate)
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         ero = cv2.dilate(tmp, kernel, iterations=1)
#         ero[erzhi < 0.001] = 0
#         ero = cv2.erode(ero, kernel, iterations=10)
#         res_root2 = os.path.join(res_root, 'ero')
#         if not os.path.exists(res_root2):
#             os.mkdir(res_root2)
#         res_path = os.path.join(res_root2, img_name)
#         cv2.imwrite(res_path, (ero * 255).astype(np.uint8))
#
#         trimap = ero + (erzhi - ero) * 0.5
#
#         # comp corr
#         x = pil_to_np(img)
#         y = pil_to_np(bg)
#         x = np_to_torch(x).cuda()
#         y = np_to_torch(y).cuda()
#         # print(x.shape)
#         # print(y.shape)
#         x_g = rgb_to_yiq(x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :])[0]  # take the Y part
#         y_g = rgb_to_yiq(y[:, 0:1, :, :], y[:, 1:2, :, :], y[:, 2:3, :, :])[0]  # take the Y part
#         # x_g = x[:, 0:1, :, :]
#         # y_g = y[:, 0:1, :, :]
#         #
#         # cov = covar(x_g.cuda(), y_g.cuda()) ** 2
#         # cov = torch_to_np(cov)[0]
#         # cv2.normalize(cov, cov, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#         # cov = cov.astype(np.uint8)
#
#         # dis = torch.sqrt(torch.mean((x - y) ** 2, dim=1, keepdim=True))
#         # mask = (dis > 0.001).type(torch.cuda.FloatTensor)
#         mask = torch.ones_like(x[:, 0:1, :, :])
#         # print(mask.shape)
#
#         # # grad
#         # grad_x1, grad_y1 = comp_grad(x)
#         # grad_x2, grad_y2 = comp_grad(y)
#         # print('gradx1.min: {:.5f}'.format(torch.min(grad_x1)))
#         # print('gradx1.max: {:.5f}'.format(torch.max(grad_x1)))
#         # grad_1 = torch.cat([grad_x1, grad_y1], dim=1)
#         # grad_2 = torch.cat([grad_x2, grad_y2], dim=1)
#
#         # # comp corr in 3 channels respectively
#         # covs = []
#         # c = x.shape[1]
#         # for i in range(c):
#         #     x_g = x[:, i:i + 1, :, :]  # take the Y part
#         #     y_g = y[:, i:i + 1, :, :]  # take the Y part
#         #     cov, _, _ = comp_GNGC(x_g, y_g, mask)
#         #     covs.append(cov)
#         # # print('gngc.shape: {}'.format(cov.shape))
#         # cov = covs[0]
#         # for i in range(1, c):
#         #     cov += covs[1]
#         # cov /= c
#
#         # # add noise to xg, yg
#         # sigma = 0.001
#         # mean = 0
#         # x_g += sigma*torch.randn(x_g.shape).cuda()+mean
#         # y_g += sigma*torch.randn(y_g.shape).cuda()+mean
#
#         cov, vv, _ = comp_GNGC(x_g, y_g, mask)
#         cov = torch_to_np(cov)[0]
#
#         # # cov + dis
#         # pad_num = (dis.shape[0] - cov.shape[0]) // 2
#         # dis = (dis - np.min(dis)) / (np.max(dis) - np.min(dis))
#         # tmp = dis[pad_num:dis.shape[0] - pad_num, pad_num:dis.shape[1] - pad_num]
#         # print('cov.min: {:.5f}'.format(np.min(cov)))
#         # print('cov.max: {:.5f}'.format(np.max(cov)))
#         # print('dis.min: {:.5f}'.format(np.min(dis)))
#         # print('dis.max: {:.5f}'.format(np.max(dis)))
#         # cov = 0.5 * cov + 0.5 * (1 - tmp)
#
#         x_g = rgb_to_yiq(x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :])[0]  # take the Y part
#         y_g = rgb_to_yiq(y[:, 0:1, :, :], y[:, 1:2, :, :], y[:, 2:3, :, :])[0]  # take the Y part
#         res_root2 = os.path.join(res_root, 'var_img')
#         if not os.path.exists(res_root2):
#             os.mkdir(res_root2)
#         res_path = os.path.join(res_root2, img_name)
#         tmp = comp_var(x_g)
#         tmp = torch_to_np(tmp)[0]
#         print('var_img.min: {:.5f}'.format(np.min(tmp)))
#         print('var_img.max: {:.5f}'.format(np.max(tmp)))
#         tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
#         cv2.imwrite(res_path, np.clip(tmp * 255, 0, 255).astype(np.uint8))
#
#         res_root2 = os.path.join(res_root, 'var_bg')
#         if not os.path.exists(res_root2):
#             os.mkdir(res_root2)
#         res_path = os.path.join(res_root2, img_name)
#         tmp = comp_var(y_g)
#         tmp = torch_to_np(tmp)[0]
#         print('var_bg.min: {:.5f}'.format(np.min(tmp)))
#         print('var_bg.max: {:.5f}'.format(np.max(tmp)))
#         tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
#         cv2.imwrite(res_path, np.clip(tmp * 255, 0, 255).astype(np.uint8))
#
#         res_root2 = os.path.join(res_root, 'vv')
#         if not os.path.exists(res_root2):
#             os.mkdir(res_root2)
#         res_path = os.path.join(res_root2, img_name)
#         tmp = torch_to_np(vv)[0]
#         print('vv.min: {:.5f}'.format(np.min(tmp)))
#         print('vv.max: {:.5f}'.format(np.max(tmp)))
#         tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
#         cv2.imwrite(res_path, np.clip(tmp * 255, 0, 255).astype(np.uint8))
#
#         res_root2 = os.path.join(res_root, 'res')
#         if not os.path.exists(res_root2):
#             os.mkdir(res_root2)
#         res_path = os.path.join(res_root2, img_name)
#         tmp = np.zeros_like(cov)
#         tmp[cov > 0] = cov[cov > 0]
#         cv2.imwrite(res_path, np.clip(tmp * 255, 0, 255).astype(np.uint8))
#
#         res_root2 = os.path.join(res_root, 'res_mult_dis')
#         if not os.path.exists(res_root2):
#             os.mkdir(res_root2)
#         res_path = os.path.join(res_root2, img_name)
#         tmp = np.zeros_like(cov)
#         tmp[cov > 0] = cov[cov > 0]
#         pad_num = (ero.shape[0] - tmp.shape[0]) // 2
#         tmp1 = np.zeros_like(ero)
#         tmp1[pad_num:ero.shape[0] - pad_num, pad_num:ero.shape[1] - pad_num] = tmp
#         res_mult_dis = (1-tmp1+0.00001) * dis
#         cv2.imwrite(res_path, np.clip(res_mult_dis * 255, 0, 255).astype(np.uint8))
#
#         res_root2 = os.path.join(res_root, 'res_mult_dis_thres')
#         if not os.path.exists(res_root2):
#             os.mkdir(res_root2)
#         res_path = os.path.join(res_root2, img_name)
#         res_mult_dis_thres = (res_mult_dis > 0.1).astype(np.uint8)
#         cv2.imwrite(res_path, np.clip(res_mult_dis_thres * 255, 0, 255))
#
#
#         print(np.min(cov))
#         print(np.max(cov))
#         # res_thres = (cov > np.min(cov) + 0.9 * (np.max(cov) - np.min(cov))).astype(np.float32)
#         res_thres = (cov > 0.8 * np.max(cov)).astype(np.float32)
#
#         res_root2 = os.path.join(res_root, 'res_thres')
#         if not os.path.exists(res_root2):
#             os.mkdir(res_root2)
#         res_path = os.path.join(res_root2, img_name)
#         cv2.imwrite(res_path, (res_thres * 255).astype(np.uint8))
#         # res_thres = res_thres.astype(np.uint8) * 255
#         # if np.sum(cov != cov) != 0:
#         #     print('nan is in cov')
#         # cv2.normalize(cov, cov, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#         # cov = cov.astype(np.uint8)
#
#         # refine trimap according to the corr
#         print('ero.shape: {}'.format(ero.shape))
#         print('res_thres.shape: {}'.format(res_thres.shape))
#         pad_num = (ero.shape[0] - res_thres.shape[0]) // 2
#         print(pad_num)
#         tmp = np.zeros_like(ero)
#         print('tmp.shape: {}'.format(tmp.shape))
#         tmp[pad_num:ero.shape[0] - pad_num, pad_num:ero.shape[1] - pad_num] = res_thres
#         area = tmp * ero
#         unknown = cv2.erode(area, kernel, iterations=1)
#         unknown = cv2.dilate(unknown, kernel, iterations=15)
#         trimap = trimap * (1 - unknown) + 0.5 * unknown
#         trimap = (trimap * 255).astype(np.uint8)
#
#         res_root2 = os.path.join(res_root, 'tri')
#         if not os.path.exists(res_root2):
#             os.mkdir(res_root2)
#         res_path = os.path.join(res_root2, img_name)
#         cv2.imwrite(res_path, trimap)


class Trimap_generator():
    def __init__(self):
        self.covar = CovarianceLayer().cuda()
        self.comp_var = VarianceLayer(channels=1).cuda()
        self.comp_GNGC = YIQGNGCLoss(shape=5).cuda()
        self.comp_mean = MeanLayer(channels=1).cuda()
        self.comp_grad = GradientLoss().cuda()

    def get_trimap(self, img, bg, img_name=None, res_root=None):
        '''comments are in Trimap_generator_real which are the same in this class except that some hyperparams'''
        # comp ori seg
        dis = np.sqrt(np.mean((img - bg) ** 2, axis=0))
        # 0.05 is smaller than Trimap_generator_real
        thres = np.min(dis) + 0.05 * (np.max(dis) - np.min(dis)) # + 0.1 * (np.max(dis) - np.min(dis))
        erzhi = np.zeros_like(dis)
        erzhi[dis > thres] = 1

        tmp = (dis > (np.min(dis) + 0.02 * (np.max(dis) - np.min(dis)))).astype(np.float32)

        # preprocess(dilate)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ero = cv2.dilate(tmp, kernel, iterations=1)
        ero[erzhi < 0.001] = 0
        ero = cv2.erode(ero, kernel, iterations=10)

        trimap = ero + (erzhi - ero) * 0.5

        # comp corr
        x = np_to_torch(img).cuda()
        y = np_to_torch(bg).cuda()
        x_g = rgb_to_yiq(x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :])[0]  # take the Y part
        y_g = rgb_to_yiq(y[:, 0:1, :, :], y[:, 1:2, :, :], y[:, 2:3, :, :])[0]  # take the Y part
        mask = torch.ones_like(x[:, 0:1, :, :])
        cov, vv, _ = self.comp_GNGC(x_g, y_g, mask)
        cov = torch_to_np(cov)[0]

        res_thres = (cov > 0.8 * np.max(cov)).astype(np.float32)


        pad_num = (ero.shape[0] - res_thres.shape[0]) // 2
        # print(pad_num)
        tmp = np.zeros_like(ero)
        # print('tmp.shape: {}'.format(tmp.shape))
        tmp[pad_num:ero.shape[0] - pad_num, pad_num:ero.shape[1] - pad_num] = res_thres
        area = tmp * ero
        unknown = cv2.erode(area, kernel, iterations=1)
        unknown = cv2.dilate(unknown, kernel, iterations=15)
        trimap = trimap * (1 - unknown) + 0.5 * unknown

        if res_root is not None:
            # res_root2 = os.path.join(res_root, 'tri')
            if not os.path.exists(res_root):
                os.mkdir(res_root)
            res_path = os.path.join(res_root, img_name)
            trimap_show = (trimap * 255).astype(np.uint8)
            cv2.imwrite(res_path, trimap_show)

        return trimap




class Trimap_generator_real():
    def __init__(self):
        self.covar = CovarianceLayer().cuda()
        self.comp_var = VarianceLayer(channels=1).cuda()
        self.comp_GNGC = YIQGNGCLoss(shape=5).cuda()
        self.comp_mean = MeanLayer(channels=1).cuda()
        self.comp_grad = GradientLoss().cuda()

    def get_trimap(self, img, bg, img_name=None, res_root=None):
        # comp ori seg map (include most possible foregrounds)
        dis = np.sqrt(np.mean((img - bg) ** 2, axis=0))
        thres = np.min(dis) + 0.1 * (np.max(dis) - np.min(dis)) # + 0.1 * (np.max(dis) - np.min(dis))
        erzhi = np.zeros_like(dis)
        erzhi[dis > thres] = 1

        # decrease the threshold to expand the foreground area
        tmp = (dis > (np.min(dis) + 0.02 * (np.max(dis) - np.min(dis)))).astype(np.float32)
        # first dilate then erode to get the initial trimap
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ero = cv2.dilate(tmp, kernel, iterations=1)
        ero[erzhi < 0.001] = 0
        # ero is foreground area in initial trimap
        ero = cv2.erode(ero, kernel, iterations=10)

        trimap = ero + (erzhi - ero) * 0.5

        # comp corr
        x = np_to_torch(img).cuda()
        y = np_to_torch(bg).cuda()
        ## transform to YIQ color space
        x_g = rgb_to_yiq(x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :])[0]  # take the Y part
        y_g = rgb_to_yiq(y[:, 0:1, :, :], y[:, 1:2, :, :], y[:, 2:3, :, :])[0]  # take the Y part
        mask = torch.ones_like(x[:, 0:1, :, :])
        ## comp covariance
        cov, vv, _ = self.comp_GNGC(x_g, y_g, mask)
        cov = torch_to_np(cov)[0]

        # semi-transparent area
        res_thres = (cov > 0.8 * np.max(cov)).astype(np.float32)

        # pad res_thres into the same shape of ero
        pad_num = (ero.shape[0] - res_thres.shape[0]) // 2
        tmp = np.zeros_like(ero)
        tmp[pad_num:ero.shape[0] - pad_num, pad_num:ero.shape[1] - pad_num] = res_thres
        # get other unknown area which covariance is high (contains background color)
        area = tmp * ero
        # erode and dilate to smooth the area
        unknown = cv2.erode(area, kernel, iterations=1)
        unknown = cv2.dilate(unknown, kernel, iterations=15)
        # expand the unknown in trimap
        trimap = trimap * (1 - unknown) + 0.5 * unknown

        # store in disk
        if res_root is not None:
            # res_root2 = os.path.join(res_root, 'tri')
            if not os.path.exists(res_root):
                os.mkdir(res_root)
            res_path = os.path.join(res_root, img_name)
            trimap_show = (trimap * 255).astype(np.uint8)
            cv2.imwrite(res_path, trimap_show)

        return trimap

from utils import prepare_image
if __name__ == '__main__':
    img_names = ['1.png', '3.png', '8.png']
    img_root = 'D:/lbl/2021hanjia/My_Paper/deepMattePrior_tcsvt_v15/main/fig/res_real_hard/Data/'
    bg_root = 'D:/lbl/2021hanjia/My_Paper/deepMattePrior_tcsvt_v15/main/fig/res_real_hard/BG/'
    store_root = 'D:/lbl/2021hanjia/My_Paper/deepMattePrior_tcsvt_v15/main/fig/res_real_hard/Ours_trimap/'
    os.makedirs(store_root, exist_ok=True)

    tri_gen = Trimap_generator_real()

    for img_name in img_names:
        img_path = os.path.join(img_root, img_name)
        img = prepare_image(img_path)
        bg_path = os.path.join(bg_root, img_name)
        bg = prepare_image(bg_path)

        trimap = tri_gen.get_trimap(img, bg, img_name, store_root)
