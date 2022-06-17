import torch
from torch import nn
import numpy as np
from torch.nn import functional
import torch.nn.functional as F
from colorsys import rgb_to_yiq
from utils import *

class GrayscaleLayer(nn.Module):
    def __init__(self):
        super(GrayscaleLayer, self).__init__()

    def forward(self, x):
        return torch.mean(x, 1, keepdim=True)

class StdLoss(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(StdLoss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)
        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(functional.conv2d(x, self.image), functional.conv2d(x, self.blur))


class ExclusionLoss(nn.Module):

    def __init__(self, level=3):
        """
        Loss on the gradient. based on:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            # alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
            # alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))
            alphay = 1
            alphax = 1
            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            # gradx_loss.append(torch.mean(((gradx1_s ** 2) * (gradx2_s ** 2))) ** 0.25)
            # grady_loss.append(torch.mean(((grady1_s ** 2) * (grady2_s ** 2))) ** 0.25)
            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(3):
            for j in range(3):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady


class ExtendedL1Loss(nn.Module):
    """
    also pays attention to the mask, to be relative to its size
    """
    def __init__(self):
        super(ExtendedL1Loss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, a, b, mask):
        # tmp = []
        # c = mask.shape[1]
        # if c > 1:
        #     for i in range(c):
        #         tmp.append(mask)
        #     mask = torch.cat(tmp, dim=1)
        normalizer = self.l1(mask, torch.zeros(mask.shape).cuda())
        # print('normalizer: {}'.format(normalizer))
        # if normalizer < 0.1:
        c = self.l1(mask * a, mask * b) / (normalizer + 0.00001)
        return c

class ExtendedL2Loss(nn.Module):
    """
    also pays attention to the mask, to be relative to its size
    """
    def __init__(self):
        super(ExtendedL2Loss, self).__init__()
        self.l1 = nn.L1Loss().cuda()
        self.l2 = nn.MSELoss().cuda()

    def forward(self, a, b, mask):
        # tmp = []
        # c = mask.shape[1]
        # if c > 1:
        #     for i in range(c):
        #         tmp.append(mask)
        #     mask = torch.cat(tmp, dim=1)
        normalizer = self.l1(mask, torch.zeros(mask.shape).cuda())
        # print('normalizer: {}'.format(normalizer))
        # if normalizer < 0.1:
        c = self.l2(mask * a, mask * b) / (normalizer + 0.00001)
        return c


class NonBlurryLoss(nn.Module):
    def __init__(self):
        """
        Loss on the distance to 0.5
        """
        super(NonBlurryLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x):
        return 1 - self.mse(x, torch.ones_like(x) * 0.5)


class GrayscaleLoss(nn.Module):
    def __init__(self):
        super(GrayscaleLoss, self).__init__()
        self.gray_scale = GrayscaleLayer()
        self.mse = nn.MSELoss().cuda()

    def forward(self, x, y):
        x_g = self.gray_scale(x)
        y_g = self.gray_scale(y)
        return self.mse(x_g, y_g)


class GrayLoss(nn.Module):
    def __init__(self):
        super(GrayLoss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, x):
        y = torch.ones_like(x) / 2.
        return 1 / self.l1(x, y)


class LocalSmoothLoss(nn.Module):
    """
    L1 loss on the gradient of the picture
    """
    def __init__(self):
        super(LocalSmoothLoss, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
        return 0.5 * torch.mean(gradient_a_x) + 0.5 * torch.mean(gradient_a_y)


def comp_grad(a):
    gradient_a_x = torch.zeros_like(a)
    gradient_a_y = torch.zeros_like(a)
    gradient_a_x[:, :, :, :-1] = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
    gradient_a_y[:, :, :-1, :] = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])

    return gradient_a_x, gradient_a_y


class GradientLoss(nn.Module):
    """
    L1 loss on the gradient of the picture
    """
    def __init__(self):
        super(GradientLoss, self).__init__()
        filter_x = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
        self.filter_x = nn.Parameter(data=torch.cuda.FloatTensor(filter_x[np.newaxis, np.newaxis, :, :]), requires_grad=False)
        filter_y = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        self.filter_y = nn.Parameter(data=torch.cuda.FloatTensor(filter_y[np.newaxis, np.newaxis, :, :]), requires_grad=False)
        assert self.filter_x.shape == self.filter_y.shape == (1, 1, 3, 3)

        patch_size = 3
        mask = np.zeros((1, 1, patch_size, patch_size))
        mask[:, :, patch_size // 2, patch_size // 2] = 1.
        self.ones_mask = nn.Parameter(data=torch.cuda.FloatTensor(mask), requires_grad=False)

        self.extendedL1Loss = ExtendedL1Loss()

    def forward(self, a, b, mask=None):
        assert a.shape == b.shape
        tmp_filter_x = []
        tmp_filter_y = []
        c = a.shape[1]
        filter_x = self.filter_x
        filter_y = self.filter_y
        if c > 1:
            for i in range(c):
                tmp_filter_x.append(self.filter_x)
                tmp_filter_y.append(self.filter_y)
            filter_x = torch.cat(tmp_filter_x, dim=1) / 3
            filter_y = torch.cat(tmp_filter_y, dim=1) / 3
        gradient_a_x = F.conv2d(a, filter_x)
        gradient_a_y = F.conv2d(a, filter_y)
        gradient_b_x = F.conv2d(b, filter_x)
        gradient_b_y = F.conv2d(b, filter_y)
        if mask is None:
            mask = torch.ones_like(gradient_a_x).cuda()
        else:
            mask = F.conv2d(mask, self.ones_mask)
        # print(gradient_a_x.shape, gradient_a_y.shape, gradient_b_x.shape, gradient_b_y.shape, mask.shape)
        return 0.5 * self.extendedL1Loss(gradient_a_x, gradient_b_x, mask) + 0.5 * self.extendedL1Loss(gradient_a_y, gradient_b_y, mask)
#
# class VarianceLayer(nn.Module):
#     # TODO: make it pad-able
#     def __init__(self, patch_size=5, channels=1):
#         self.patch_size = patch_size
#         super(VarianceLayer, self).__init__()
#         mean_mask = np.ones((channels, channels, patch_size, patch_size)) / (patch_size * patch_size)
#         self.mean_mask = nn.Parameter(data=torch.cuda.FloatTensor(mean_mask), requires_grad=False)
#         mask = np.zeros((channels, channels, patch_size, patch_size))
#         mask[:, :, patch_size // 2, patch_size // 2] = 1.
#         self.ones_mask = nn.Parameter(data=torch.cuda.FloatTensor(mask), requires_grad=False)
#
#     def forward(self, x):
#         Ex_E = F.conv2d(x, self.ones_mask) - F.conv2d(x, self.mean_mask)
#         return F.conv2d((Ex_E) ** 2, self.mean_mask)
#
#
# class CovarianceLayer(nn.Module):
#     def __init__(self, patch_size=5, channels=1):
#         self.patch_size = patch_size
#         super(CovarianceLayer, self).__init__()
#         mean_mask = np.ones((channels, channels, patch_size, patch_size)) / (patch_size * patch_size)
#         self.mean_mask = nn.Parameter(data=torch.cuda.FloatTensor(mean_mask), requires_grad=False)
#         mask = np.zeros((channels, channels, patch_size, patch_size))
#         mask[:, :, patch_size // 2, patch_size // 2] = 1.
#         self.ones_mask = nn.Parameter(data=torch.cuda.FloatTensor(mask), requires_grad=False)
#
#     def forward(self, x, y):
#         return F.conv2d((F.conv2d(x, self.ones_mask) - F.conv2d(x, self.mean_mask)) *
#                         (F.conv2d(y, self.ones_mask) - F.conv2d(y, self.mean_mask)), self.mean_mask)

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
#             y_g = rgb_to_yiq(y[:, 0:1, :, :], y[:, 1:2, :, :], y[:, 2:3, :, :])[0]   # take the Y part
#         else:
#             assert x.shape[1] == 1
#             x_g = x  # take the Y part
#             y_g = y  # take the Y part
#         c = torch.mean(self.covar(x_g, y_g) ** 2)
#         vv = torch.mean(self.var(x_g) * self.var(y_g))
#         return c / vv

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