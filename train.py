from collections import namedtuple

from net import *
from loss import StdLoss, YIQGNGCLoss, GradientLoss, ExtendedL1Loss, GrayLoss, comp_grad
from noise import get_noise, NoiseNet
from utils import *
import os
import time
import math
import torch
from torch import nn
# from net.downsampler import *
# from skimage.measure import compare_psnr
# from cv2.ximgproc import guidedFilter

SegmentationResult = namedtuple("SegmentationResult", ['mask', 'learned_mask', 'left', 'right', 'psnr'])

def format_second(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    string = '{:d}:{:d}:{:d}'.format(h, m, s)
    return string

class Segmentation(object):
    def __init__(self, image_name, image, bg, plot_root, record_path, gt_alpha=None, plot_during_training=True,
                 zero_step_iter_num=1500,
                 first_step_iter_num=2000,
                 second_step_iter_num=4000,
                 bg_hint=None, fg_hint=None,
                 show_every=500):
        '''

        :param image_name: the name of the input image
        :param image: input image
        :param bg: input bg
        :param plot_root: the root the will store the results
        :param record_path: the path of txt that store the metrics
        :param gt_alpha: gt alpha if exists to compute the metrics
        :param plot_during_training: whether print the learning process and whether store the intermediate results
        :param zero_step_iter_num: the iteration numbers of the zero step
        :param first_step_iter_num: the iteration numbers of the first step
        :param second_step_iter_num: the iteration numbers of the second step
        :param bg_hint: initial bg area
        :param fg_hint: initial fg area
        :param show_every: the interval of iterations to store the results
        '''
        self.image_name = image_name
        self.image = image
        self.bg = bg
        self.gt_alpha = gt_alpha
        self.bg_hint = bg_hint
        self.fg_hint = fg_hint
        self.image_torch = None
        self.bg_torch = None

        self.plot_during_training = plot_during_training
        self.plot_root = plot_root
        self.record_path = record_path

        self.left_net = None
        self.mask_net = None

        self.input_depth = 2
        self.left_net_input = None
        self.mask_net_input = None
        self.left_net_output = None
        self.mask_net_output = None

        self.parameters = None
        self.learning_rate = 0.001
        self.total_loss = None
        self.losses_str = ''

        self.show_every = show_every
        self.second_step_done = False
        self.zero_step_iter_num = zero_step_iter_num
        self.first_step_iter_num = first_step_iter_num
        self.second_step_iter_num = second_step_iter_num
        self.third_step_iter_num = 1000
        self.current_psnr = None
        self.current_alphamse = None
        self.current_alphasad = None
        self.best_alphamse = 10000.
        self.best_iter = -1
        self.t1 = 0

        self.mp_fg = 500
        self.mp_mask = 500
        self.mp_fg_ori = 1000
        self.mp_mask_ori = 1000

        self.stop_fg = 500
        self.stop_mask = 500
        self.stop_fg_ori = 500
        self.stop_mask_ori = 500

        fg_bg_path = os.path.join(plot_root, 'fg_bg/')
        if not os.path.exists(fg_bg_path):
            os.mkdir(fg_bg_path)
        self.fg_bg_path = fg_bg_path

        alpha_path = os.path.join(plot_root, 'alpha/')
        if not os.path.exists(alpha_path):
            os.mkdir(alpha_path)
        self.alpha_path = alpha_path


        pred_img_path = os.path.join(plot_root, 'pred_img/')
        if not os.path.exists(pred_img_path):
            os.mkdir(pred_img_path)
        self.pred_img_path = pred_img_path

        other_path = os.path.join(plot_root, 'other/')
        if not os.path.exists(other_path):
            os.mkdir(other_path)
        self.other_path = other_path

        self._init_all()

    def _init_nets(self):
        pad = 'reflection'
        left_net = dip_segment(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 0],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')


        self.left_net = left_net.type(torch.cuda.FloatTensor)


        mask_net = dip_segment(
            self.input_depth, 1,
            num_channels_skip=[0, 0, 0],
            filter_size_down=3,
            filter_size_up=3,
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')


        self.mask_net = mask_net.type(torch.cuda.FloatTensor)

    def _init_images(self):
        self.image_torch = np_to_torch(self.image).type(torch.cuda.FloatTensor)
        self.bg_torch = np_to_torch(self.bg).type(torch.cuda.FloatTensor)
        if self.bg_hint is not None:
            assert self.bg_hint.shape[1:] == self.image.shape[1:], (self.bg_hint.shape[1:], self.image.shape[1:])
            self.bg_hint_torch = np_to_torch(self.bg_hint).type(torch.cuda.FloatTensor)
        else:
            self.bg_hint = None
        if self.fg_hint is not None:
            assert self.fg_hint.shape[1:] == self.image.shape[1:], (self.fg_hint.shape[1:], self.image.shape[1:])
            self.fg_hint_torch = np_to_torch(self.fg_hint).type(torch.cuda.FloatTensor)
        else:
            self.fg_hint = None

        self.trimap = self.fg_hint + (1 - self.fg_hint - self.bg_hint) * 0.5
        self.trimap_torch = np_to_torch(self.trimap).type(torch.cuda.FloatTensor)
        self.trimap_mask = 1 - (self.fg_hint + self.bg_hint)
        self.trimap_mask_torch = np_to_torch(self.trimap_mask).type(torch.cuda.FloatTensor)

        self.fg_tmp = self.image_torch

        self._init_weight()
        self._init_fg_weight()

    def _init_noise(self):
        input_type = 'noise'
        self.left_net_input = get_noise(self.input_depth,
                                          input_type,
                                           (self.image_torch.shape[2], self.image_torch.shape[3])).type(torch.cuda.FloatTensor).detach()

        input_type = 'noise'
        self.mask_net_input = get_noise(self.input_depth,
                                          input_type,
                                          (self.image_torch.shape[2], self.image_torch.shape[3])).type(torch.cuda.FloatTensor).detach()


    def _init_parameters(self):
        self.parameters = [p for p in self.left_net.parameters()] + \
                          [p for p in self.mask_net.parameters()]
                          # [p for p in self.right_net.parameters()]

    def _init_losses(self):
        data_type = torch.cuda.FloatTensor
        self.gngc_loss = YIQGNGCLoss().type(data_type)
        self.l2_loss = nn.MSELoss().type(data_type)
        self.l1_loss = nn.L1Loss().type(data_type)
        self.extended_l1_loss = ExtendedL1Loss().type(data_type)
        # self.blur_function = StdLoss().type(data_type)
        self.gradient_loss = GradientLoss().type(data_type)
        self.gray_loss = GrayLoss().type(data_type)

    def _init_all(self):
        self._init_images()
        self._init_losses()
        self._init_nets()
        self._init_parameters()
        self._init_noise()

    def _post_init(self, low, high):
        self.fg_hint_torch = torch.zeros_like(self.mask_net_output).type(torch.cuda.FloatTensor)
        self.fg_hint_torch[self.mask_net_output > high] = 1
        # self.bg_hint_torch = torch.zeros_like(self.mask_net_output).type(torch.cuda.FloatTensor)
        # self.bg_hint_torch[self.mask_net_output < low] = 1
        self.fg_hint = torch_to_np(self.fg_hint_torch)
        # self.bg_hint = torch_to_np(self.bg_hint_torch)

        self.trimap = self.fg_hint + (1 - self.fg_hint - self.bg_hint) * 0.5
        self.trimap_torch = np_to_torch(self.trimap).type(torch.cuda.FloatTensor)
        self.trimap_mask = 1 - (self.fg_hint + self.bg_hint)
        self.trimap_mask_torch = np_to_torch(self.trimap_mask).type(torch.cuda.FloatTensor)

    def _init_weight(self):
        weight_decay = 0.5
        weight_matrix = weight_decay * self.trimap_mask[0].copy()
        print('weight_matrix.shape: {}'.format(weight_matrix.shape))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for i in range(5):
            weight_decay -= 0.1
            exist = (weight_matrix>0).astype(np.float32)
            ero = cv2.dilate(exist, kernel, iterations=10)
            weight_matrix += (ero - exist) * weight_decay
        self.weight_matrix = weight_matrix[np.newaxis, :, :]
        assert np.all(self.weight_matrix >= 0)
        self.weight_matrix_torch = np_to_torch(self.weight_matrix).type(torch.cuda.FloatTensor)

    def _init_fg_weight(self):
        weight_decay = 1
        weight_matrix = weight_decay * self.fg_hint[0].copy()
        print('weight_matrix.shape: {}'.format(weight_matrix.shape))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ero = (weight_matrix>0).astype(np.float32)
        for i in range(5):
            weight_decay -= 0.2
            exist = (ero>0).astype(np.float32)
            ero = cv2.erode(exist, kernel, iterations=10)
            weight_matrix -= (exist - ero) * weight_decay
        self.fg_weight_matrix = weight_matrix[np.newaxis, :, :] * 2
        assert np.all(weight_matrix <= 1)
        self.fg_weight_matrix_torch = np_to_torch(self.fg_weight_matrix).type(torch.cuda.FloatTensor)

    def _post_init_new(self, low, high):
        self.fg_hint_torch = torch.zeros_like(self.mask_net_output).type(torch.cuda.FloatTensor)
        self.fg_hint_torch[(self.mask_net_output + self.trimap_torch) / 2 > high] = 1
        # self.bg_hint_torch = torch.zeros_like(self.mask_net_output).type(torch.cuda.FloatTensor)
        # self.bg_hint_torch[(self.mask_net_output + self.trimap_torch) / 2 < low] = 1
        self.fg_hint = torch_to_np(self.fg_hint_torch)
        # self.bg_hint = torch_to_np(self.bg_hint_torch)
        self.trimap = self.fg_hint + (1 - self.fg_hint - self.bg_hint) * 0.5
        self.trimap_torch = np_to_torch(self.trimap).type(torch.cuda.FloatTensor)
        self.trimap_mask = 1 - (self.fg_hint + self.bg_hint)
        self.trimap_mask_torch = np_to_torch(self.trimap_mask).type(torch.cuda.FloatTensor)


        self._init_weight()
        self._init_fg_weight()


    def hyper_update_stop(self):
        psnr = self.compute_psnr(torch_to_np(self.left_net_output), self.image, self.fg_hint)
        right = 25
        left = 15
        ratio = min(max((psnr - left) / (right - left), 0), 1)
        self.stop_mask = max(self.stop_fg_ori * ratio // 100 * 100, 1)
        self.stop_fg = self.stop_fg_ori


        with open(self.record_path, 'a+') as f:
            s = 'psnr: {:.5f}  stop_fg: {}  stop_mask: {}\n'.format(psnr, self.stop_fg, self.stop_mask)
            f.write(s)


    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        self.t1 = time.time()

        # first run
        # # step 1
        step = 0
        accu = 0
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.zero_step_iter_num):
            optimizer.zero_grad()
            self._step1_optimization_closure(step, j)
            self._finalize_iteration(accu + j)
            optimizer.step()
            if self.plot_during_training:
                self._iteration_plot_closure(accu + j)
        self._step_plot_closure(step)
        self._plot_first(step)

        # # step 2
        step += 1
        accu += self.zero_step_iter_num
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.first_step_iter_num):
            optimizer.zero_grad()
            self._step2_optimization_closure(step, j)
            self._finalize_iteration(accu + j)
            # if self.second_step_done:
            #     break
            optimizer.step()
            if self.plot_during_training:
                self._iteration_plot_closure(accu + j)
        self._step_plot_closure(step)
        # get new initialization
        self._post_init_new(0.27, 0.73)
        self._plot_first(step)

        # second run
        # # step 1
        step += 1
        accu += self.first_step_iter_num
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.zero_step_iter_num):
            optimizer.zero_grad()
            self._step1_optimization_closure(step, j)
            self._finalize_iteration(accu + j)
            optimizer.step()
            if self.plot_during_training:
                self._iteration_plot_closure(accu + j)
        self._step_plot_closure(step)
        self._plot_first(step)

        # # step 2
        step += 1
        accu += self.zero_step_iter_num
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.second_step_iter_num):
            optimizer.zero_grad()
            self._step2_optimization_closure(step, j)
            self._finalize_iteration(accu + j)
            # if self.second_step_done:
            #     break
            optimizer.step()
            if self.plot_during_training:
                self._iteration_plot_closure(accu + j)
        self._step_plot_closure(step)
        self._post_init(0.05, 0.95)
        self._plot_first(step)

        # # step 3 (projection)
        step += 1
        accu += self.second_step_iter_num
        self.hyper_update_stop()
        optimizer = torch.optim.Adam([
            {'params': self.left_net.parameters(), 'lr': self.learning_rate},
            {'params': self.mask_net.parameters(), 'lr': self.learning_rate*5}])
        lambda1 = lambda iter: 1
        lambda2 = lambda iter: 0.9 ** (iter // 100)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        self.m = (self.mask_net_output.detach() > 0.99).float()
        self.fg_tmp = self.left_net_output.detach()
        assert self.fg_tmp is not self.left_net_output
        for j in range(self.third_step_iter_num):
            optimizer.zero_grad()
            self._step5_optimization_closure(step, j)
            self._finalize_iteration(accu + j)
            # if self.second_step_done:
            #     break
            optimizer.step()
            if self.plot_during_training:
                self._iteration_plot_closure(accu + j)
            scheduler.step(j)
        self._step_plot_closure(step)
        self._post_init(0.05, 0.95)
        self._plot_first(step)


        return self.best_alphamse, self.current_alphamse, self.current_alphasad

    def finalize_first_step(self):
        left = torch_to_np(self.left_net_outputs[0])
        right = torch_to_np(self.right_net_outputs[0])
        save_image(self.image_name + "_1_left", left)
        save_image(self.image_name + "_1_right", right)
        save_image(self.image_name + "_hint1", self.bg_hint)
        save_image(self.image_name + "_hint2", self.fg_hint)
        save_image(self.image_name + "_hint1_masked", self.bg_hint * self.image)
        save_image(self.image_name + "_hint2_masked", self.fg_hint * self.image)

    def finalize(self):
        save_image(self.image_name + "_left", self.best_result.left)
        save_image(self.image_name + "_learned_mask", self.best_result.learned_mask)
        save_image(self.image_name + "_right", self.best_result.right)
        save_image(self.image_name + "_original", self.images[0])
        # save_image(self.image_name + "_fg_bg", ((self.fg_hint - self.bg_hint) + 1) / 2)
        save_image(self.image_name + "_mask", self.best_result.mask)

    def _plot_first(self, step):
        plot_image_grid('image', self.image, output_path=self.pred_img_path)
        # plot_image_grid('trimap', self.trimap, output_path=self.other_path)
        plot_image_grid('trimap_{}'.format(step), self.trimap, output_path=self.other_path)
        plot_image_grid('fg_masked_{}'.format(step), self.fg_hint * torch_to_np(self.fg_tmp), output_path=self.other_path)
        plot_image_grid('fg_hint_{}'.format(step), self.fg_hint, output_path=self.other_path)
        plot_image_grid('bg_hint_{}'.format(step), self.bg_hint, output_path=self.other_path)
        plot_image_grid('bg', self.bg, output_path=self.other_path)
        plot_image_grid('weight_matrix_{}'.format(step), self.weight_matrix, output_path=self.other_path)
        plot_image_grid('fg_weight_matrix_{}'.format(step), self.fg_weight_matrix, output_path=self.other_path)
        plot_image_grid('fg_maskedsmall_{}'.format(step), (self.fg_weight_matrix>0.99).astype(np.float32) * torch_to_np(self.fg_tmp), output_path=self.other_path)

        if self.gt_alpha is not None:
            plot_image_grid('gt_alpha', self.gt_alpha, output_path=self.alpha_path)
            # plot_image_grid('gt_alpha', self.gt_alpha, output_path=self.masked_alpha_path)



    def _initialize_step1(self, step, iteration):
        self._initialize_any_step(step, iteration)

    def _initialize_step2(self, step, iteration):
        self._initialize_any_step(step, iteration)

    def _initialize_any_step0(self, step, iteration):
        if step == 0:
            tmp = 1000
        else:
            tmp = 1000

        if iteration > self.second_step_iter_num:
            reg_noise_std_mask = 0
        elif iteration < tmp:
            reg_noise_std_mask = (1 / 1000.) * (iteration // 100)
            # if step == 2:
            #     ratio = torch.sum(self.trimap_mask_torch) / torch.sum(torch.ones_like(self.trimap_mask_torch))
            #     max_std = 0.01 * ratio
            #     reg_noise_std_left = max(min(reg_noise_std_left, max_std), 1 / 1000.)
            #     reg_noise_std_right = reg_noise_std_mask = reg_noise_std_left
        else:
            reg_noise_std_mask = 1 / 1000.

        if iteration > self.second_step_iter_num:
            reg_noise_std_left = 0
        elif iteration < tmp:
            reg_noise_std_left = (1 / 1000.) * (iteration // 100)
            # if step == 2:
            #     ratio = torch.sum(self.trimap_mask_torch) / torch.sum(torch.ones_like(self.trimap_mask_torch))
            #     max_std = 0.01 * ratio
            #     reg_noise_std_left = max(min(reg_noise_std_left, max_std), 1 / 1000.)
            #     reg_noise_std_right = reg_noise_std_mask = reg_noise_std_left

        else:
            reg_noise_std_left = 1 / 1000.


        # creates left_net_inputs and right_net_inputs by adding small noise
        left_net_input = self.left_net_input + (self.left_net_input.clone().normal_() * reg_noise_std_left)
        # right_net_input = self.right_net_input + (self.right_net_input.clone().normal_() * reg_noise_std_right)
        mask_net_input = self.mask_net_input + (self.mask_net_input.clone().normal_() * reg_noise_std_mask)
        # applies the nets
        self.left_net_output = self.left_net(left_net_input)
        # self.right_net_output = self.right_net(right_net_input)
        self.mask_net_output = self.mask_net(mask_net_input)
        # if self.mask_net_output is not None:
        #     self.mask_net_output = 0.4 * self.mask_net_output.detach() + 0.6 * self.mask_net(mask_net_input)
        # else:
        #     self.mask_net_output = self.mask_net(mask_net_input)
        self.total_loss = 0
        self.losses_str = ''

    def _initialize_any_step(self, step, iteration):

        if step == 1:
            max_iteration = self.second_step_iter_num #self.first_step_iter_num
        else:
            max_iteration = self.second_step_iter_num

        if iteration >= max_iteration - self.stop_mask:
            reg_noise_std_mask = 0
        # elif step == 1 and iteration == self.first_step_iter_num - 1:
        #     reg_noise_std_mask = 0
        elif iteration < self.mp_mask:
            reg_noise_std_mask = (1 / 1000.) * (iteration // 100)
        else:
            reg_noise_std_mask = 1 / 1000.

        if iteration >= max_iteration - self.stop_fg:
            reg_noise_std_left = 0
        # elif step == 1 and iteration == self.first_step_iter_num - 500:
        #     reg_noise_std_left = 0
        elif iteration < self.mp_fg:
            reg_noise_std_left = (1 / 1000.) * (iteration // 100)
        else:
            reg_noise_std_left = 1 / 1000.

        # creates left_net_inputs and right_net_inputs by adding small noise
        left_net_input = self.left_net_input + (self.left_net_input.clone().normal_() * reg_noise_std_left)
        # right_net_input = self.right_net_input + (self.right_net_input.clone().normal_() * reg_noise_std_right)
        mask_net_input = self.mask_net_input + (self.mask_net_input.clone().normal_() * reg_noise_std_mask)
        # applies the nets
        if not (iteration >= self.mp_fg and iteration < self.mp_mask):
            self.left_net_output = self.left_net(left_net_input)
        self.mask_net_output = self.mask_net(mask_net_input)
        self.total_loss = 0
        self.losses_str = ''

    def _step5_optimization_closure(self, step, iteration):
        if iteration >= self.third_step_iter_num - 1:
            reg_noise_std_left = 0.
        else:
            reg_noise_std_left = 1 / 1000.

        if iteration >= self.third_step_iter_num - self.stop_mask:
            reg_noise_std_mask = 0.
        else:
            reg_noise_std_mask = 1 / 1000.



        # creates left_net_inputs and right_net_inputs by adding small noise
        left_net_input = self.left_net_input + (self.left_net_input.clone().normal_() * reg_noise_std_left)
        mask_net_input = self.mask_net_input + (self.mask_net_input.clone().normal_() * reg_noise_std_mask)
        # applies the nets
        self.left_net_output = self.left_net(left_net_input)
        self.mask_net_output = self.mask_net(mask_net_input)

        self.total_loss = 0
        self.losses_str = ''


        left_out = self.image_torch * self.m + self.left_net_output * (1 - self.m)
        pred_img = self.mask_net_output * left_out + (1 - self.mask_net_output) * self.bg_torch
        reconst_loss = self.l1_loss(pred_img, self.image_torch)
        self.total_loss += reconst_loss
        self.losses_str += ' reconst_loss: {:.5f}'.format(reconst_loss)

        self.pred_img = pred_img



        self.total_loss.backward()  #(retain_graph=True)




    def _step1_optimization_closure(self, step, iteration):
        """
        the real iteration is step * self.num_iter_per_step + iteration
        :param iteration:
        :param step:
        :return:
        """
        self._initialize_any_step0(step, iteration)
        if np.any(self.fg_hint > 0.999) or np.any(self.bg_hint > 0.999):
            self._step1_optimize_with_hints(step, iteration)
        else:
            self._step1_optimize_without_hints(iteration)

    def _step2_optimization_closure(self, step, iteration):
        """
        the real iteration is step * self.num_iter_per_step + iteration
        :param iteration:
        :param step:
        :return:
        """
        self._initialize_step2(step, iteration)
        if np.any(self.fg_hint > 0.999) or np.any(self.bg_hint > 0.999):
            self._step2_optimize_with_hints(step, iteration)
        else:
            self._step2_optimize_without_hints(iteration)

    def _step0_optimization_closure(self, iteration):
        """
        the real iteration is step * self.num_iter_per_step + iteration
        :param iteration:
        :param step:
        :return:
        """
        self._initialize_any_step0(0, iteration)
        # if np.any(self.fg_hint > 0.999) or np.any(self.bg_hint > 0.999):
        #     self._step2_optimize_with_hints(iteration)
        # else:
        #     self._step2_optimize_without_hints(iteration)
        self._step0_optimize(iteration)

    def _step1_optimize_without_hints(self, iteration):
        near05_loss = self.l1_loss(self.mask_net_output, torch.ones_like(self.mask_net_output) / 2)
        self.total_loss += near05_loss
        self.losses_str += ' near05_loss: {:.5f}'.format(near05_loss)

        self.total_loss.backward()  #(retain_graph=True)

    def _step1_optimize_with_hints(self, step, iteration):
        """
        optimization, where hints are given
        :param iteration:
        :return:
        """
        if step == 0:
            weight = self.fg_weight_matrix_torch
        else:
            weight = self.fg_hint_torch
        if np.any(self.fg_hint > 0.999):
            fg_loss = self.extended_l1_loss(self.left_net_output, self.fg_tmp, weight)
            self.total_loss += fg_loss
            self.losses_str += ' fg_loss: {:.5f}'.format(fg_loss)
        # if np.any(self.bg_hint > 0.999):
        #     bg_loss = self.extended_l1_loss(self.right_net_output, self.image_torch, self.bg_hint_torch)
        #     self.total_loss += bg_loss
        #     self.losses_str += ' bg_loss: {:.5f}'.format(bg_loss)

        # print(torch.max(self.mask_net_output))
        # print(torch.max(self.trimap_torch))
        tri_loss = self.l1_loss(self.mask_net_output, self.trimap_torch)
        # tri_loss = torch.mean(torch.abs(self.trimap_torch - self.mask_net_output))
        self.total_loss += tri_loss
        self.losses_str += ' tri_loss: {:.5f}'.format(tri_loss)

        self.total_loss.backward() #(retain_graph=True)

        self.pred_img = self.left_net_output * self.mask_net_output + self.bg_torch * (1 - self.mask_net_output)

    def _step0_optimize(self, iteration):
        pred_img = self.mask_net_output * self.left_net_output + (1 - self.mask_net_output) * self.bg_torch
        reconst_loss = self.l1_loss(pred_img, self.image_torch)
        self.total_loss += reconst_loss
        self.losses_str += ' reconst_loss: {:.5f}'.format(reconst_loss)

        # # gngc loss
        # gngc_loss = self.gngc_loss(self.left_net_output, self.bg_torch)
        # self.total_loss += gngc_loss
        # self.losses_str += ' gngc_loss: {:.5f}'.format(gngc_loss)

        if np.any(self.fg_hint > 0.999):
            fg_loss = self.extended_l1_loss(self.left_net_output, self.fg_tmp, self.fg_hint_torch)
            self.total_loss += fg_loss
            self.losses_str += ' fg_loss: {:.5f}'.format(fg_loss)

        # if iteration < 1000:
        tri_loss = self.extended_l1_loss(self.mask_net_output, self.fg_hint_torch, self.fg_hint_torch)
        self.total_loss += tri_loss
        self.losses_str += ' tri_loss: {:.5f}'.format(tri_loss)


        self.total_loss.backward() #(retain_graph=True)

    def _step2_optimize_without_hints(self, iteration):
        weight = torch.ones_like(self.trimap_mask_torch)
        weight[self.trimap_mask_torch < 0.001] = 0.5
        pred_img = self.mask_net_output * self.left_net_output + (1 - self.mask_net_output) * self.right_net_output
        reconst_loss = self.l1_loss(pred_img * weight, self.image_torch * weight)
        self.total_loss += reconst_loss
        self.losses_str += ' reconst_loss: {:.5f}'.format(reconst_loss)

        # gngc loss
        gngc_loss = self.gngc_loss(self.left_net_output, self.right_net_output)
        self.total_loss += gngc_loss
        self.losses_str += ' gngc_loss: {:.5f}'.format(gngc_loss)

        # gradient consistency
        grad_loss = 0.1 * self.gradient_loss(pred_img, self.image_torch, self.trimap_mask_torch)
        self.total_loss += grad_loss
        self.losses_str += ' grad_loss: {:.5f}'.format(grad_loss)

        # self.current_gradient = self.gradient_loss(mask_out)
        # self.total_loss += (0.01 * (iteration // 100)) * self.current_gradient
        self.total_loss.backward() #(retain_graph=True)

    def _step2_optimize_with_hints(self, step, iteration):

        pred_img = self.mask_net_output * self.left_net_output + (1 - self.mask_net_output) * self.bg_torch
        # pred_img = self.mask_net_output * self.left_net_output + (1 - self.mask_net_output) * self.bg_torch
        reconst_loss = self.l1_loss(pred_img, self.image_torch)
        # if step == 1: #iteration >= self.mp_mask and iteration >= self.mp_fg:
        #     reconst_loss = self.extended_l1_loss(pred_img, self.image_torch, self.weight_matrix_torch+0.5)
        # else:
        #     reconst_loss = self.l1_loss(pred_img, self.image_torch)
        self.total_loss += reconst_loss
        self.losses_str += ' reconst_loss: {:.5f}'.format(reconst_loss)

        self.pred_img = pred_img


        self.total_loss.backward() #(retain_graph=True)

    def fba_fusion(self, alpha, img, F, B):
        F = ((alpha * img + (1 - alpha ** 2) * F - alpha * (1 - alpha) * B))
        # B = ((1 - alpha) * img + (2 * alpha - alpha**2) * B - alpha * (1 - alpha) * F)

        F = torch.clamp(F, 0, 1)
        # B = np.clip(B, 0, 1)
        la = 0.1

        alpha = (alpha * la + torch.sum((img - B) * (F - B), 1, keepdim=True)) / (
                torch.sum((F - B) * (F - B), 1, keepdim=True) + la)
        alpha = torch.clamp(alpha, 0, 1)
        return alpha, F, B

    def compute_mse(self, a, b, mask=None):
        # inputs are numpy
        assert a.shape == b.shape
        if mask is None:
            mask = np.ones([1, a.shape[1], a.shape[2]])

        mse = np.sum(((a - b)*mask) ** 2)
        normalizer = np.sum(mask) + 0.00001
        # print(mse, normalizer)
        mse = mse / normalizer

        return mse

    def compute_psnr(self, a, b, mask=None):
        assert a.shape == b.shape
        if mask is None:
            mask = np.ones([1, a.shape[1], a.shape[2]])

        mse = self.compute_mse(a, b, mask)
        psnr = 10 * np.log10(1. / mse)

        return psnr

    def compute_sad(self, pd, gt, mask=None):
        assert pd.shape == gt.shape
        if mask is None:
            mask = np.ones([1, pd.shape[1], pd.shape[2]])
        error_map = np.abs(pd - gt)
        loss = np.sum(error_map * mask)
        # the loss is scaled by 1000 due to the large images
        loss = loss / 1000
        return loss

    def _finalize_iteration(self, iteration):
        left_out_np = torch_to_np(self.left_net_output)
        right_out_np = self.bg #torch_to_np(self.right_net_output)
        original_image = self.image
        mask_out_np = torch_to_np(self.mask_net_output)
        pred_img = torch_to_np(self.pred_img) #mask_out_np * left_out_np + (1 - mask_out_np) * right_out_np
        self.current_psnr = self.compute_psnr(original_image, pred_img)
        if self.gt_alpha is not None:
            self.current_alphamse = self.compute_mse(mask_out_np, self.gt_alpha)
            self.current_alphasad = self.compute_sad(mask_out_np, self.gt_alpha)
            if self.current_alphamse < self.best_alphamse:
                self.best_alphamse = self.current_alphamse
                self.best_iter = iteration
        # TODO: run only in the second step
        if self.current_psnr > 32:
            self.second_step_done = True

    def _iteration_plot_closure(self, iter_number):
        last_time = time.time() - self.t1
        speed = last_time / (iter_number + 1)
        extra_time = speed * (self.zero_step_iter_num * 2 + self.first_step_iter_num + self.second_step_iter_num - iter_number - 1)
        # speed_str = format_second(speed)
        extra_time_str = format_second(extra_time)
        if self.gt_alpha is not None:
            print('Iteration {:d} alpha_mse {:.5f} alpha_sad {:.5f} PSNR {:.5f} total_loss {:.5f} {} speed {:.5f}s/iter extra_time {}'.format(
                iter_number, self.current_alphamse, self.current_alphasad, self.current_psnr, self.total_loss.item(),
                self.losses_str, speed, extra_time_str))
        else:
            print('Iteration {:d} PSNR {:.5f} total_loss {:.5f} {} speed {:.5f}s/iter extra_time {}'.format(
                iter_number, self.current_psnr, self.total_loss.item(), self.losses_str, speed, extra_time_str))
        if iter_number % self.show_every == 0: #self.show_every - 1:
            self._plot_with_name(iter_number)

    def _step_plot_closure(self, step_number):
        """
        runs at the end of each step
        :param step_number:
        :return:
        """
        self._plot_with_name("step_{}".format(step_number))

        if step_number == 4:
            left_out = self.image_torch * self.m + self.left_net_output * (1 - self.m)
            plot_image_grid("masked_fg_{}".format(step_number), np.clip(torch_to_np(left_out), 0, 1),
                            output_path=self.other_path)

        if self.gt_alpha is not None:
            s = '{} step {} img_psnr: {:.5f} best_mse: {:.5f} best_iter: {} current mse: {:.5f} current sad: {:.5f}\n'.format(
                self.image_name, step_number, self.current_psnr, self.best_alphamse, self.best_iter,
                self.current_alphamse, self.current_alphasad
            )
            with open(self.record_path, 'a+') as f:
                f.write(s)


    def _plot_with_name(self, name):
        plot_image_grid("{}_left".format(name), np.clip(torch_to_np(self.left_net_output), 0, 1),
                        output_path=self.fg_bg_path)
        plot_image_grid("{}_right".format(name), np.clip(torch_to_np(self.bg_torch), 0, 1),
                        output_path=self.fg_bg_path)

        mask_out_np = torch_to_np(self.mask_net_output)
        plot_image_grid("{}_learned_mask".format(name), np.clip(mask_out_np, 0, 1),
                        output_path=self.alpha_path)

        # masked_mask = mask_out_np * self.trimap_mask + self.trimap * (1 - self.trimap_mask)
        # plot_image_grid("{}_masked_mask".format(name), np.clip(masked_mask, 0, 1),
        #                 output_path=self.masked_alpha_path)


        pred_img = np.clip(torch_to_np(self.pred_img), 0, 1)#np.clip(mask_out_np * torch_to_np(self.left_net_output) + (1 - mask_out_np) * torch_to_np(self.bg_torch), 0, 1)
        plot_image_grid("{}_learned_image".format(name), pred_img, output_path=self.pred_img_path)

def get_names(img_root, except_root, group_num):
    # get non-portrait img names
    # img_root = 'D:/lbl/myBackgroundMattingData/merged_test_all_512/img/'
    # except_root = 'D:/lbl/myBackgroundMattingData/merged_test20_20_512/img/'

    all_img_names = os.listdir(img_root)
    except_names = os.listdir(except_root)
    except_names_20 = []
    for i, img_name in enumerate(except_names):
        if i % 20 == group_num:
            except_names_20.append(img_name)
    img_names_20 = []
    for i, img_name in enumerate(all_img_names):
        if i % 20 == group_num and img_name not in except_names_20:
            img_names_20.append(img_name)

    print(len(img_names_20))
    print(img_names_20)
    return img_names_20



DATA_ROOT = "ROOT_TO_INPUT"
RESULT_ROOT = "ROOT_TO_RESULT"
testImgroot = DATA_ROOT + "DIR_TO_INPUT_IMAGE"
testBgroot = DATA_ROOT + "DIR_TO_INPUT_BG"
testAlpharoot = None # "DIR_TO_GT_ALPHA" IF EXISTS TO COMPUTE SAD AND MSE
# THE INPUT TRIMAPS ARE GENERATED BY initialization.py
testTriroot = DATA_ROOT + "DIR_TO_INPUT_TRI"
res_root = RESULT_ROOT + "DIR_TO_RESULT"


if __name__ == '__main__':
    img_names = os.listdir(testImgroot)

    os.makedirs(res_root, exist_ok=True)
    record_path = os.path.join(res_root, 'record.txt')

    aver_best_mse = 0
    aver_curr_mse = 0
    aver_curr_sad = 0
    num = 0

    for i, img_name in enumerate(img_names):
        t0 = time.time()
        if True: #i % 20 == 0:
            print(img_name)
            num += 1
            # read input
            img_path = os.path.join(testImgroot, img_name)
            I = prepare_image(img_path)
            # print(I.shape)
            # print(np.max(I))
            bg_path = os.path.join(testBgroot, img_name)
            bg = prepare_image(bg_path)
            # trimap generated by initialization.py
            tri_path = os.path.join(testTriroot, img_name)
            trimap = prepare_image(tri_path)

            if trimap.shape[0] > 1:
                trimap = trimap[0:1]
            # print(np.max(trimap))
            gt_alpha = None
            if testAlpharoot is not None:
                alpha_path = os.path.join(testAlpharoot, img_name)
                gt_alpha = prepare_image(alpha_path)[0:1]
            # print(bg.shape)
            # print(gt_alpha.shape)
            # print(np.max(gt_alpha))

            # known area of fg and bg
            fg_hint = np.zeros_like(trimap)
            bg_hint = np.zeros_like(trimap)
            fg_hint[trimap > 0.999] = 1
            bg_hint[trimap < 0.001] = 1
            # trimap_mask = 1 - (fg_hint + bg_hint)

            # store dir
            plot_root = os.path.join(res_root, img_name.split('.')[0])
            if not os.path.exists(plot_root):
                os.mkdir(plot_root)

            # init model
            seg = Segmentation(img_name, I, bg, plot_root, record_path, gt_alpha, plot_during_training=True,
                               zero_step_iter_num=10, first_step_iter_num=2000, second_step_iter_num=4000,
                               fg_hint=fg_hint, bg_hint=bg_hint,
                               show_every=100)

            # begin to learn
            best_alphamse, curr_alphamse, curr_alphasad = seg.optimize()

            if testAlpharoot is not None:
                aver_best_mse += best_alphamse
                aver_curr_mse += curr_alphamse
                aver_curr_sad += curr_alphasad

                aver_best_mse = aver_best_mse / num  # len(img_names)
                aver_curr_mse = aver_curr_mse / num  # len(img_names)
                aver_curr_sad = aver_curr_sad / num  # len(img_names)

        t = time.time() - t0
        print('time: {}'.format(format_second(t)))


    # record if gt alpha is known
    with open(record_path, 'a+') as f:
        s = 'aver_best_mse: {:.5f} aver_curr_mse: {:.5f} aver_curr_sad: {:.5f}\n'.format(aver_best_mse, aver_curr_mse, aver_curr_sad)
        f.write(s)
