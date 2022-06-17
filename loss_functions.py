import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import matplotlib.pyplot as plt
import pdb
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
#import scipy.io as sio
from colorsys import rgb_to_yiq


class alpha_loss(_Loss):
    def __init__(self):
        super(alpha_loss,self).__init__()

    def forward(self,alpha,alpha_pred,mask):
        return normalized_l1_loss(alpha,alpha_pred,mask)




class compose_loss(_Loss):
    def __init__(self):
        super(compose_loss,self).__init__()

    def forward(self,image,alpha_pred,fg,bg,mask):

        alpha_pred=(alpha_pred+1)/2

        comp=fg*alpha_pred + (1-alpha_pred)*bg

        return normalized_l1_loss(image,comp,mask)

# class alpha_gradient_loss(_Loss):
# 	def __init__(self):
# 		super(alpha_gradient_loss,self).__init__()
#
# 	def forward(self,alpha,alpha_pred,mask):
#
# 		fx = torch.Tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]]); fx=fx.view((1,1,3,3)); fx=Variable(fx.cuda())
# 		fy = torch.Tensor([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]); fy=fy.view((1,1,3,3)); fy=Variable(fy.cuda())
#
# 		G_x = F.conv2d(alpha,fx,padding=1); G_y = F.conv2d(alpha,fy,padding=1)
# 		G_x_pred = F.conv2d(alpha_pred,fx,padding=1); G_y_pred = F.conv2d(alpha_pred,fy,padding=1)
#
# 		loss=normalized_l1_loss(G_x,G_x_pred,mask) + normalized_l1_loss(G_y,G_y_pred,mask)
#
# 		return loss

class alpha_gradient_loss(_Loss):
    def __init__(self):
        super(alpha_gradient_loss,self).__init__()

    def forward(self,img,img_pred,mask):
        c = img.shape[1]

        fx = torch.Tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]]); fx=fx.view((1,1,3,3)); fx=Variable(fx.cuda())
        fy = torch.Tensor([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]); fy=fy.view((1,1,3,3)); fy=Variable(fy.cuda())

        fx = torch.cat([fx]*c, dim=0)
        fy = torch.cat([fy]*c, dim=0)
        # print('fx.shape: {}'.format(fx.shape))
        # print('fy.shape: {}'.format(fy.shape))

        G_x = F.conv2d(img,fx,padding=1,groups=c); G_y = F.conv2d(img,fy,padding=1,groups=c)
        G_x_pred = F.conv2d(img_pred,fx,padding=1,groups=c); G_y_pred = F.conv2d(img_pred,fy,padding=1,groups=c)

        loss=normalized_l1_loss(G_x,G_x_pred,mask) + normalized_l1_loss(G_y,G_y_pred,mask)

        return loss

# class alpha_gradient_reg_loss(_Loss):
# 	def __init__(self):
# 		super(alpha_gradient_reg_loss,self).__init__()
#
# 	def forward(self,alpha,mask):
#
# 		fx = torch.Tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]]); fx=fx.view((1,1,3,3)); fx=Variable(fx.cuda())
# 		fy = torch.Tensor([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]); fy=fy.view((1,1,3,3)); fy=Variable(fy.cuda())
#
# 		G_x = F.conv2d(alpha,fx,padding=1); G_y = F.conv2d(alpha,fy,padding=1)
#
# 		loss=(torch.sum(torch.abs(G_x))+torch.sum(torch.abs(G_y)))/torch.sum(mask)
#
# 		return loss

class alpha_gradient_reg_loss(_Loss):
    def __init__(self):
        super(alpha_gradient_reg_loss,self).__init__()

    def forward(self,img,mask):
        eps = 1e-6
        c = img.shape[1]

        fx = torch.Tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]]); fx=fx.view((1,1,3,3)); fx=Variable(fx.cuda())
        fy = torch.Tensor([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]); fy=fy.view((1,1,3,3)); fy=Variable(fy.cuda())

        fx = torch.cat([fx] * c, dim=0)
        fy = torch.cat([fy] * c, dim=0)

        G_x = F.conv2d(img,fx,padding=1,groups=c); G_y = F.conv2d(img,fy,padding=1,groups=c)

        loss=(torch.sum(torch.abs(G_x*mask))+torch.sum(torch.abs(G_y*mask)))/(torch.sum(mask)+eps)
        # print('torch.sum(torch.abs(G_x*mask)): {}'.format(torch.sum(torch.abs(G_x*mask))))
        # print('torch.sum(mask): {}'.format(torch.sum(mask)))

        return loss, G_x, G_y

class get_gradient(nn.Module):
    def __init__(self):
        super(get_gradient,self).__init__()

    def forward(self,alpha):
        # print('alpha.min: {}  alpha.max: {}'.format(torch.min(alpha), torch.max(alpha)))

        fx = torch.Tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]]); fx=fx.view((1,1,3,3)); fx=Variable(fx.cuda())
        fy = torch.Tensor([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]); fy=fy.view((1,1,3,3)); fy=Variable(fy.cuda())

        G_x = F.conv2d(alpha,fx,padding=1); G_y = F.conv2d(alpha,fy,padding=1)
        # print('G_x.shape: {}'.format(G_x.shape))
        # print('G_x.min: {}  G_x.max: {}'.format(torch.min(G_x), torch.max(G_x)))

        grad = torch.sqrt(G_x**2 + G_y**2)

        return grad

class GANloss(_Loss):
    def __init__(self):
        super(GANloss,self).__init__()

    def forward(self,pred,label_type):
        MSE=nn.MSELoss()

        loss=0
        for i in range(0,len(pred)):
            if label_type:
                labels=torch.ones(pred[i][0].shape)
            else:
                labels=torch.zeros(pred[i][0].shape)
            labels=Variable(labels.cuda())

            loss += MSE(pred[i][0],labels)

        return loss/len(pred)



def normalized_l1_loss(alpha,alpha_pred,mask):
    loss=0; eps=1e-6;
    for i in range(alpha.shape[0]):
        if mask[i,...].sum()>0:
            loss = loss + torch.sum(torch.abs(alpha[i,...]*mask[i,...]-alpha_pred[i,...]*mask[i,...]))/(torch.sum(mask[i,...])+eps)
    loss=loss/alpha.shape[0]

    return loss

class GradientLoss(nn.Module):
    """
    L1 loss on the gradient of the picture
    """
    def __init__(self):
        super(GradientLoss, self).__init__()
        filter_x1 = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
        self.filter_x1 = nn.Parameter(data=torch.cuda.FloatTensor(filter_x1[np.newaxis, np.newaxis, :, :]), requires_grad=False)
        filter_x2 = np.array([[-1, -2, -1],[2, 4, 2],[-1, -2, -1]])
        self.filter_x2 = nn.Parameter(data=torch.cuda.FloatTensor(filter_x2[np.newaxis, np.newaxis, :, :]), requires_grad=False)
        filter_y1 = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        self.filter_y1 = nn.Parameter(data=torch.cuda.FloatTensor(filter_y1[np.newaxis, np.newaxis, :, :]), requires_grad=False)
        filter_y2 = np.array([[-1, 2, -1],[-2, 4, -2],[-1, 2, -1]])
        self.filter_y2 = nn.Parameter(data=torch.cuda.FloatTensor(filter_y2[np.newaxis, np.newaxis, :, :]), requires_grad=False)
        assert self.filter_x1.shape == self.filter_y1.shape == (1, 1, 3, 3)
        assert self.filter_x2.shape == self.filter_y2.shape == (1, 1, 3, 3)

        patch_size = 3
        mask = np.zeros((1, 1, patch_size, patch_size))
        mask[:, :, patch_size // 2, patch_size // 2] = 1.
        self.ones_mask = nn.Parameter(data=torch.cuda.FloatTensor(mask), requires_grad=False)


    def forward(self, a, mode=1):
        tmp_filter_x = []
        tmp_filter_y = []
        c = a.shape[1]
        if mode == 1:
            filter_x = self.filter_x1
            filter_y = self.filter_y1
        else:
            filter_x = self.filter_x2
            filter_y = self.filter_y2
        if c > 1:
            for i in range(c):
                tmp_filter_x.append(filter_x)
                tmp_filter_y.append(filter_y)
            filter_x = torch.cat(tmp_filter_x, dim=1) / 3
            filter_y = torch.cat(tmp_filter_y, dim=1) / 3
        gradient_a_x = F.conv2d(a, filter_x)
        gradient_a_y = F.conv2d(a, filter_y)
        scale_x = max(torch.abs(torch.min(gradient_a_x)), torch.abs(torch.max(gradient_a_x)))
        scale_y = max(torch.abs(torch.min(gradient_a_y)), torch.abs(torch.max(gradient_a_y)))
        gradient_a_x /= scale_x
        gradient_a_y /= scale_y

        pad_num = (a.shape[2] - gradient_a_x.shape[2]) // 2
        grad_x1_l = torch.zeros((1, 1, a.shape[2], a.shape[3])).cuda()
        grad_x1_l[:, :, pad_num:a.shape[2] - pad_num, pad_num:a.shape[3] - pad_num] = gradient_a_x
        grad_y1_l = torch.zeros((1, 1, a.shape[2], a.shape[3])).cuda()
        grad_y1_l[:, :, pad_num:a.shape[2] - pad_num, pad_num:a.shape[3] - pad_num] = gradient_a_y

        # print(gradient_a_x.shape, gradient_a_y.shape, gradient_b_x.shape, gradient_b_y.shape, mask.shape)
        return grad_x1_l, grad_y1_l

class MeanLayer(nn.Module):
    # TODO: make it pad-able
    def __init__(self, patch_size=5, channels=1):
        self.patch_size = patch_size
        super(MeanLayer, self).__init__()
        mean_mask = np.ones((channels, channels, patch_size, patch_size)) / (patch_size * patch_size)
        self.mean_mask = nn.Parameter(data=torch.cuda.FloatTensor(mean_mask), requires_grad=False)

    def forward(self, x):
        return F.conv2d(x, self.mean_mask)

class VarianceLayer(nn.Module):
    # TODO: make it pad-able
    def __init__(self, patch_size=5, channels=1):
        self.patch_size = patch_size
        super(VarianceLayer, self).__init__()
        mean_mask = np.ones((channels, channels, patch_size, patch_size)) / (patch_size * patch_size)
        self.mean_mask = nn.Parameter(data=torch.cuda.FloatTensor(mean_mask), requires_grad=False)
        mask = np.zeros((channels, channels, patch_size, patch_size))
        mask[:, :, patch_size // 2, patch_size // 2] = 1.
        self.ones_mask = nn.Parameter(data=torch.cuda.FloatTensor(mask), requires_grad=False)

    def forward(self, x):
        Ex_E = F.conv2d(x, self.ones_mask) - F.conv2d(x, self.mean_mask)
        return F.conv2d((Ex_E) ** 2, self.mean_mask)
        # return F.conv2d(x ** 2, self.mean_mask) - F.conv2d(x, self.mean_mask) ** 2
        # return F.conv2d(x ** 2, self.mean_mask)

class CovarianceLayer(nn.Module):
    def __init__(self, patch_size=5, channels=1):
        self.patch_size = patch_size
        super(CovarianceLayer, self).__init__()
        mean_mask = np.ones((channels, channels, patch_size, patch_size)) / (patch_size * patch_size)
        self.mean_mask = nn.Parameter(data=torch.cuda.FloatTensor(mean_mask), requires_grad=False)
        mask = np.zeros((channels, channels, patch_size, patch_size))
        mask[:, :, patch_size // 2, patch_size // 2] = 1.
        self.ones_mask = nn.Parameter(data=torch.cuda.FloatTensor(mask), requires_grad=False)

    def forward(self, x, y):
        return F.conv2d((F.conv2d(x, self.ones_mask) - F.conv2d(x, self.mean_mask)) *
                        (F.conv2d(y, self.ones_mask) - F.conv2d(y, self.mean_mask)), self.mean_mask)
        # return F.conv2d(x * y, self.mean_mask) - F.conv2d(x, self.mean_mask) * F.conv2d(y, self.mean_mask)
        # return F.conv2d(x * y, self.mean_mask)


# class YIQGNGCLoss(nn.Module):
#     def __init__(self, shape=5):
#         super(YIQGNGCLoss, self).__init__()
#         self.shape = shape
#         self.var = VarianceLayer(self.shape, channels=1)
#         self.covar = CovarianceLayer(self.shape, channels=1)
#
#     def forward(self, x, y):
#         if x.shape[1] == 3:
#             x_g = rgb_to_yiq(x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :])[0]  # take the Y part
#             y_g = rgb_to_yiq(y[:, 0:1, :, :], y[:, 1:2, :, :], y[:, 2:3, :, :])[0]  # take the Y part
#         else:
#             assert x.shape[1] == 1
#             x_g = x  # take the Y part
#             y_g = y  # take the Y part
#         c = torch.mean(self.covar(x_g, y_g) ** 2)
#         # c = self.covar(x_g, y_g) ** 2
#         vv = torch.mean(self.var(x_g) * self.var(y_g))
#         return c / vv

# class YIQGNGCLoss(nn.Module):
#     def __init__(self, shape=5):
#         super(YIQGNGCLoss, self).__init__()
#         self.shape = shape
#         self.var = VarianceLayer(self.shape, channels=1)
#         self.covar = CovarianceLayer(self.shape, channels=1)
#
#     def forward(self, x, y, mask):
#         eps = 0.000001
#         if x.shape[1] == 3:
#             x_g = rgb_to_yiq(x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :])[0]  # take the Y part
#             y_g = rgb_to_yiq(y[:, 0:1, :, :], y[:, 1:2, :, :], y[:, 2:3, :, :])[0]  # take the Y part
#         else:
#             assert x.shape[1] == 1
#             x_g = x  # take the Y part
#             y_g = y  # take the Y part
#         # c = torch.mean(self.covar(x_g, y_g) ** 2)
#         c = self.covar(x_g, y_g)
#
#         # c = self.covar(x_g, y_g) ** 2
#         # pad_num_x = (mask.shape[2] - c.shape[2]) // 2
#         # pad_num_y = (mask.shape[3] - c.shape[3]) // 2
#         # print(pad_num_x, pad_num_y)
#         # h, w = mask.shape[2:]
#         # mask = mask[:, :, pad_num_x:h - pad_num_x, pad_num_y:w - pad_num_y]
#         # print('mask.shape: {}'.format(mask.shape))
#         # c = c * mask
#         # vv = torch.mean(self.var(x_g) * self.var(y_g))
#         # vv = torch.mean(self.var(x_g) * self.var(y_g) * mask)
#         vv = torch.sqrt(self.var(x_g) * self.var(y_g))
#         vv[vv < 0.000001] = 0.000001
#         corr = c / vv
#         # corr[vv < 0.000001] = 0
#         print('corr.min: {:.5f}'.format(torch.min(corr)))
#         print('corr.max: {:.5f}'.format(torch.max(corr)))
#         corr[corr < -1] = -1
#         corr[corr > 1] = 1
#
#         return corr, vv, c

class YIQGNGCLoss(nn.Module):
    def __init__(self, shape=5):
        super(YIQGNGCLoss, self).__init__()
        self.shape = shape
        self.var = VarianceLayer(self.shape, channels=1)
        self.covar = CovarianceLayer(self.shape, channels=1)

    def forward(self, x, y, mask):
        eps = 0.000001
        if x.shape[1] == 3:
            x_g = rgb_to_yiq(x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :])[0]  # take the Y part
            y_g = rgb_to_yiq(y[:, 0:1, :, :], y[:, 1:2, :, :], y[:, 2:3, :, :])[0]  # take the Y part
        else:
            assert x.shape[1] == 1
            x_g = x  # take the Y part
            y_g = y  # take the Y part
        # c = torch.mean(self.covar(x_g, y_g) ** 2)
        c = self.covar(x_g, y_g)

        # c = self.covar(x_g, y_g) ** 2
        # pad_num_x = (mask.shape[2] - c.shape[2]) // 2
        # pad_num_y = (mask.shape[3] - c.shape[3]) // 2
        # print(pad_num_x, pad_num_y)
        # h, w = mask.shape[2:]
        # mask = mask[:, :, pad_num_x:h - pad_num_x, pad_num_y:w - pad_num_y]
        # print('mask.shape: {}'.format(mask.shape))
        # c = c * mask
        # vv = torch.mean(self.var(x_g) * self.var(y_g))
        # vv = torch.mean(self.var(x_g) * self.var(y_g) * mask)
        vv = torch.sqrt(self.var(x_g) * self.var(y_g))
        c[vv < 0.000001] = 0
        vv[vv < 0.000001] = 0.000001
        corr = c / vv
        # corr[vv < 0.000001] = 0
        print('corr.min: {:.5f}'.format(torch.min(corr)))
        print('corr.max: {:.5f}'.format(torch.max(corr)))
        print('c.min: {:.5f}'.format(torch.min(c)))
        print('c.max: {:.5f}'.format(torch.max(c)))
        print('vv.min: {:.5f}'.format(torch.min(vv)))
        print('vv.max: {:.5f}'.format(torch.max(vv)))
        corr[corr < 0] = 0
        corr[corr > 1] = 1
        # corr[(self.var(x_g) < 0.001) | (self.var(y_g) < 0.001)] = 0

        return corr, vv, c