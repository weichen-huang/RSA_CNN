import random
import numpy as np

from copy import deepcopy
import fileio as fio
from dnnbrain_lib import dnn_mask, dnn_fe, array_statistic
from dnnbrain_lib import UnivariateMapping, MultivariateMapping

import abc
import cv2
import time
import copy
import torch
import numpy as np

from os import remove
from os.path import join as pjoin
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from torch.optim import Adam
import torch.nn as nn
from torch.nn.functional import interpolate
from matplotlib import pyplot as plt
from dnnbrain_lib import ip, array_statistic
from skimage import filters, segmentation
from skimage.color import rgb2gray
from skimage.morphology import convex_hull_image, erosion, square
from torch.autograd import Variable
from collections import OrderedDict

import numpy as np

from scipy.signal import convolve
from nipy.modalities.fmri.hemodynamic_models import spm_hrf


def convolve_hrf(X, onsets, durations, n_vol, tr, ops=100):
    """
    Convolve each X's column iteratively with HRF and align with the timeline of BOLD signal

    Parameters
    ----------
    X : array
        Shape = (n_event, n_sample)
    onsets : array_like
        In sec. size = n_event
    durations : array_like
        In sec. size = n_event
    n_vol : int
        The number of volumes of BOLD signal
    tr : float
        Repeat time in second
    ops : int
        Oversampling number per second

    Returns
    -------
    X_hrfed : array
        The result after convolution and alignment
    """
    assert np.ndim(X) == 2, 'X must be a 2D array'
    assert X.shape[0] == len(onsets) and X.shape[0] == len(durations), \
        'The length of onsets and durations should be matched with the number of events.'
    assert ops in (10, 100, 1000), 'Oversampling rate must be one of the (10, 100, 1000)!'

    # unify the precision
    decimals = int(np.log10(ops))
    onsets = np.round(np.asarray(onsets), decimals=decimals)
    durations = np.round(np.asarray(durations), decimals=decimals)
    tr = np.round(tr, decimals=decimals)

    n_clipped = 0  # the number of clipped time points earlier than the start point of response
    onset_min = onsets.min()
    if onset_min > 0:
        # The earliest event's onset is later than the start point of response.
        # We supplement it with zero-value event to align with the response.
        X = np.insert(X, 0, np.zeros(X.shape[1]), 0)
        onsets = np.insert(onsets, 0, 0, 0)
        durations = np.insert(durations, 0, onset_min, 0)
        onset_min = 0
    elif onset_min < 0:
        print("The earliest event's onset is earlier than the start point of response.\n"
              "We clip the earlier time points after hrf_convolution to align with the response.")
        n_clipped = int(-onset_min * ops)

    # do convolution in batches for trade-off between speed and memory
    batch_size = int(100000 / ops)
    bat_indices = np.arange(0, X.shape[-1], batch_size)
    bat_indices = np.r_[bat_indices, X.shape[-1]]

    vol_t = (np.arange(n_vol) * tr * ops).astype(int)  # compute volume acquisition timing
    n_time_point = int(((onsets + durations).max() - onset_min) * ops)
    X_hrfed = np.zeros([n_vol, 0])
    for idx, bat_idx in enumerate(bat_indices[:-1]):
        X_bat = X[:, bat_idx:bat_indices[idx + 1]]
        # generate X raw time course
        X_tc = np.zeros((n_time_point, X_bat.shape[-1]), dtype=np.float32)
        for i, onset in enumerate(onsets):
            onset_start = int(onset * ops)
            onset_end = int(onset_start + durations[i] * ops)
            X_tc[onset_start:onset_end, :] = X_bat[i, :]

        # generate hrf kernel
        hrf = spm_hrf(tr, oversampling=tr * ops)
        hrf = hrf[:, np.newaxis]

        # convolve X raw time course with hrf kernal
        X_tc_hrfed = convolve(X_tc, hrf, method='fft')
        X_tc_hrfed = X_tc_hrfed[n_clipped:, :]

        # downsample to volume timing
        X_hrfed = np.c_[X_hrfed, X_tc_hrfed[vol_t, :]]

        print('hrf convolution: sample {0} to {1} finished'.format(bat_idx + 1, bat_indices[idx + 1]))

    return X_hrfed

class Algorithm(abc.ABC):
    """
    An Abstract Base Classes class to define interface for dnn algorithm
    """

    def __init__(self, dnn, layer=None, channel=None):
        """
        Parameters
        ----------
        dnn : DNN
            A dnnbrain's DNN object.
        layer : str
            Name of the layer where the algorithm performs on.
        channel : int
            Number of the channel where the algorithm performs on.
        """
        if np.logical_xor(layer is None, channel is None):
            raise ValueError("layer and channel must be used together!")
        if layer is not None:
            self.set_layer(layer, channel)
        self.dnn = dnn
        self.dnn.eval()

    def set_layer(self, layer, channel):
        """
        Set layer or its channel.

        Parameters
        ----------
        layer : str
            Name of the layer where the algorithm performs on.
        channel : int
            Number of the channel where the algorithm performs on
            algorithm only support one channel operation at one time.
        """
        self.mask = Mask()
        self.mask.set(layer, channels=[channel])

    def get_layer(self):
        """
        Get layer or its channel

        Parameters
        ----------
        layer : str
            Name of the layer where the algorithm performs on.
        channel : int
            Number of the channel where the algorithm performs on.
        """
        layer = self.mask.layers[0]
        channel = self.mask.get(layer)['chn'][0]
        return layer, channel


class SaliencyImage(Algorithm):
    """
    An Abstract Base Classes class to define interfaces for gradient back propagation.
    Note: the saliency image values are not applied with absolute operation.
    """

    def __init__(self, dnn, from_layer=None, from_chn=None):
        """
        Parameters
        ----------
        dnn : DNN
            A dnnbrain's DNN object.
        from_layer : str
            Name of the layer where gradients back propagate from.
        from_chn : int
            umber of the channel where gradient back propagate from.
        """
        super(SaliencyImage, self).__init__(dnn, from_layer, from_chn)

        self.to_layer = None
        self.activation = None
        self.gradient = None
        self.hook_handles = []

    @abc.abstractmethod
    def register_hooks(self):
        """
        Define register hooks and register them to specific layer and channel.
        As this a abstract method, it is needed to be override in every subclass.
        """

    def backprop(self, image, to_layer=None):
        """
        Compute gradients of the to_layer corresponding to the from_layer and from_channel
        by back propagation algorithm.

        Parameters
        ---------
        image : ndarray, Tensor, PIL.Image
            Image data.
        to_layer : str
            Name of the layer where gradients back propagate to.
            If is None, get the first layer in the layers recorded in DNN.

        Return
        ------
        gradient : ndarray
            Gradients of the to_layer with shape as (n_chn, n_row, n_col).
            If layer is the first layer of the model, its shape is (3, n_height, n_width).
        """
        # register hooks
        self.to_layer = self.dnn.layers[0] if to_layer is None else to_layer
        self.register_hooks()

        # forward
        image = self.dnn.test_transform(ip.to_pil(image))
        image = image.unsqueeze(0)
        image.requires_grad_(True)
        self.dnn(image)
        # zero grads
        self.dnn.model.zero_grad()
        # backward
        self.activation.backward()
        # tensor to ndarray
        # [0] to get rid of the first dimension (1, n_chn, n_row, n_col)
        gradient = self.gradient.data.numpy()[0]

        # remove hooks
        for hook_handle in self.hook_handles:
            hook_handle.remove()

        # renew some attributions
        self.activation = None
        self.gradient = None

        return gradient

    def backprop_smooth(self, image, n_iter, sigma_multiplier=0.1, to_layer=None):
        """
        Compute smoothed gradient.
        It will use the gradient method to compute the gradient and then smooth it

        Parameters
        ----------
        image : ndarray, Tensor, PIL.Image
            Image data
        n_iter : int
            The number of noisy images to be generated before average.
        sigma_multiplier : int
            Multiply when calculating std of noise.
        to_layer : str
            Name of the layer where gradients back propagate to.
            If is None, get the first layer in the layers recorded in DNN.

        Return
        ------
        gradient : ndarray
            Gradients of the to_layer with shape as (n_chn, n_row, n_col).
            If layer is the first layer of the model, its shape is (n_chn, n_height, n_width).
        """
        assert isinstance(n_iter, int) and n_iter > 0, \
            'The number of iterations must be a positive integer!'

        # register hooks
        self.to_layer = self.dnn.layers[0] if to_layer is None else to_layer
        self.register_hooks()

        image = self.dnn.test_transform(ip.to_pil(image))
        image = image.unsqueeze(0)
        gradient = 0
        sigma = sigma_multiplier * (image.max() - image.min()).item()
        for iter_idx in range(1, n_iter + 1):
            # prepare image
            image_noisy = image + image.normal_(0, sigma ** 2)
            image_noisy.requires_grad_(True)

            # forward
            self.dnn(image_noisy)
            # clean old gradients
            self.dnn.model.zero_grad()
            # backward
            self.activation.backward()
            # tensor to ndarray
            # [0] to get rid of the first dimension (1, n_chn, n_row, n_col)
            gradient += self.gradient.data.numpy()[0]
            print(f'Finish: noisy_image{iter_idx}/{n_iter}')

        # remove hooks
        for hook_handle in self.hook_handles:
            hook_handle.remove()

        # renew some attributions
        self.activation = None
        self.gradient = None

        gradient = gradient / n_iter
        return gradient


class VanillaSaliencyImage(SaliencyImage):
    """
    A class to compute vanila Backprob gradient for a image.
    """

    def register_hooks(self):
        """
        Override the abstract method from BackPropGradient class to
        define a specific hook for vanila backprop gradient.
        """
        from_layer, from_chn = self.get_layer()

        def from_layer_acti_hook(module, feat_in, feat_out):
            self.activation = torch.mean(feat_out[0, from_chn - 1])

        def to_layer_grad_hook(module, grad_in, grad_out):
            self.gradient = grad_in[0]

        # register forward hook to the target layer
        from_module = self.dnn.layer2module(from_layer)
        from_handle = from_module.register_forward_hook(from_layer_acti_hook)
        self.hook_handles.append(from_handle)

        # register backward to the first layer
        to_module = self.dnn.layer2module(self.to_layer)
        to_handle = to_module.register_backward_hook(to_layer_grad_hook)
        self.hook_handles.append(to_handle)


class GuidedSaliencyImage(SaliencyImage):
    """
    A class to compute Guided Backprob gradient for a image.
    """

    def register_hooks(self):
        """
        Override the abstract method from BackPropGradient class to
        define a specific hook for guided backprop gradient.
        """
        from_layer, from_chn = self.get_layer()

        def from_layer_acti_hook(module, feat_in, feat_out):
            self.activation = torch.mean(feat_out[0, from_chn - 1])

        def to_layer_grad_hook(module, grad_in, grad_out):
            self.gradient = grad_in[0]

        def relu_grad_hook(module, grad_in, grad_out):
            grad_in[0][grad_out[0] <= 0] = 0

        # register hook for from_layer
        from_module = self.dnn.layer2module(from_layer)
        handle = from_module.register_forward_hook(from_layer_acti_hook)
        self.hook_handles.append(handle)

        # register backward hook to all relu layers util from_layer
        for module in self.dnn.model.modules():
            # register hooks for relu
            if isinstance(module, torch.nn.ReLU):
                handle = module.register_backward_hook(relu_grad_hook)
                self.hook_handles.append(handle)

            if module is from_module:
                break

        # register hook for to_layer
        to_module = self.dnn.layer2module(self.to_layer)
        handle = to_module.register_backward_hook(to_layer_grad_hook)
        self.hook_handles.append(handle)


class SynthesisImage(Algorithm):
    """
    Generate a synthetic image that maximally activates a neuron.
    """

    def __init__(self, dnn, layer=None, channel=None,
                 activ_metric='mean', regular_metric=None, regular_lambda=None,
                 precondition_metric=None, GB_radius=None, smooth_metric=None, factor=None):
        """

        Parameters
        ----------
        dnn : DNN
            A dnnbrain's dnn object
        layer : str
            Name of the layer where the algorithm performs on
        channel : int
            Number of the channel where the algorithm performs on
        activ_metric : str
            The metric method to summarize activation
        regular_metric : str
            The metric method of regularization
        regular_lambda : float
            The lambda of the regularization.
        precondition_metric : str
            The metric method of precondition
        GB_radius : float
            Radius parameter for 'GB', gaussian blur.
        smooth_metric : str
            The metric method of smoothing.
        factor : float
            Factor parameter for 'Fourier', smooth fourier.
        """
        super(SynthesisImage, self).__init__(dnn, layer, channel)
        self.set_loss(activ_metric, regular_metric, regular_lambda)
        self.set_precondition(precondition_metric, GB_radius)
        self.set_smooth_gradient(smooth_metric, factor)
        self.activ_loss = None
        self.optimal_image = None

        # loss recorder
        self.activ_losses = []
        self.regular_losses = []

    def set_loss(self, activ_metric, regular_metric, regular_lambda):
        """
        This method is to set loss function for optimization.
        As the target usually is a 2-D feature map in convolutional layer with multiple units,
        'active_metric' can make algorithm clear on how to omputue the loss value.
        Also there are some popular regularization to make synthesis more interpretable,
        'regular_metric' can set one of them and 'regular_lambda' give the weights of this term.

        Parameters
        ----------
        activ_metric : str
            The metric method to summarize activation.
        regular_metric : str
            The metric method of regularization.
        regular_lambda : float
            The lambda of the regularization.
        """
        # activation metric setting
        if activ_metric == 'max':
            self.activ_metric = torch.max
        elif activ_metric == 'mean':
            self.activ_metric = torch.mean
        else:
            raise AssertionError('Only max and mean activation metrics are supported')

        # regularization metric setting
        if regular_metric is None:
            self.regular_metric = lambda: 0
        elif regular_metric == 'L1':
            self.regular_metric = self._L1_norm
        elif regular_metric == 'L2':
            self.regular_metric = self._L2_norm
        elif regular_metric == 'TV':
            self.regular_metric = self._total_variation
        else:
            raise AssertionError('Only L1, L2, and total variation are supported!')

        # regularization hyperparameter setting
        self.regular_lambda = regular_lambda

    def set_precondition(self, precondition_metric, GB_radius):
        """
        This is the method to set whether a precondition metric will be used,
        precondition is one of the method to smooth the high frequency noise on
        synthesized image. It will applied on every interval image during the iteration.

        Parameters
        ----------
        precondition_metric : str
            The metric method of preconditioning.
        GB_radius : float
            Radius parameter for 'GB', gaussian blur.
        """

        # precondition metric setting
        if precondition_metric is None:
            self.precondition_metric = lambda x, y: None
        elif precondition_metric == 'GB':
            self.precondition_metric = self._gaussian_blur
        else:
            raise AssertionError('Only Gaussian Blur is supported!')
        self.GB_radius = GB_radius

    def set_smooth_gradient(self, smooth_metric, factor):
        """
        This method is to set smooth gradient metric, it's a very effective way to
        prove synthesized image quality.

        Parameters
        ----------
        smooth_metric : str
            The metric method of smoothing.
        factor : float
            Factor parameter for 'Fourier', smooth fourier.
        """
        # smooth metric setting
        if smooth_metric is None:
            self.smooth_metric = lambda x: None
        elif smooth_metric == 'Fourier':
            self.smooth_metric = self._smooth_fourier
        else:
            raise AssertionError('Only Fourier Smooth is supported!')
        self.factor = factor

    def _L1_norm(self):
        reg = torch.abs(self.optimal_image).sum()
        self.regular_losses.append(reg.item())
        return reg

    def _L2_norm(self):
        reg = torch.sqrt(torch.sum(self.optimal_image ** 2))
        self.regular_losses.append(reg.item())
        return reg

    def _total_variation(self):
        # calculate the difference of neighboring pixel-values
        diff1 = self.optimal_image[0, :, 1:, :] - self.optimal_image[0, :, :-1, :]
        diff2 = self.optimal_image[0, :, :, 1:] - self.optimal_image[0, :, :, :-1]

        # calculate the total variation
        reg = torch.sum(torch.abs(diff1)) + torch.sum(torch.abs(diff2))
        self.regular_losses.append(reg.item())
        return reg

    def _gaussian_blur(self, radius, lr):
        precond_image = filters.gaussian(self.optimal_image[0].detach().numpy(), radius)
        self.optimal_image = ip.to_tensor(precond_image).float().unsqueeze(0)
        self.optimal_image.requires_grad_(True)
        self.optimizer = Adam([self.optimal_image], lr=lr)

    def _smooth_fourier(self, factor):
        """
        Tones down the optimal image gradient with 1/sqrt(f) filter in the Fourier domain.
        Equivalent to low-pass filtering in the spatial domain.

        Parameters
        ----------
        factor : float
            Parameters used in fourier transform.
        """
        # initialize grad
        grad = self.optimal_image.grad
        # handle special situations
        if factor == 0:
            pass
        else:
            # get information of grad
            h, w = grad.size()[-2:]
            tw = np.minimum(np.arange(0, w), np.arange(w - 1, -1, -1), dtype=np.float32)
            th = np.minimum(np.arange(0, h), np.arange(h - 1, -1, -1), dtype=np.float32)
            # filtering in the spatial domain
            t = 1 / np.maximum(1.0, (tw[None, :] ** 2 + th[:, None] ** 2) ** (factor))
            F = grad.new_tensor(t / t.mean()).unsqueeze(-1)
            pp = torch.rfft(grad.data, 2, onesided=False)
            # adjust the optimal_image grad after Fourier transform
            self.optimal_image.grad = copy.copy(torch.irfft(pp * F, 2, onesided=False))

    def register_hooks(self, unit=None):
        """
        Define register hook and register them to specific layer and channel.

        Parameters
        ----------
        unit : tuple
            Determine unit position, `None` means channel, default None.

        """
        layer, chn = self.get_layer()

        def forward_hook(module, feat_in, feat_out):
            if unit is None:
                self.activ_loss = - self.activ_metric(feat_out[0, chn - 1])
            else:
                if isinstance(unit, tuple) and len(unit) == 2:
                    row = int(unit[0])
                    column = int(unit[1])
                    self.activ_loss = -feat_out[0, chn - 1, row, column]  # single unit
                else:
                    raise AssertionError('Check unit must be 2-dimensinal tuple')
            self.activ_losses.append(self.activ_loss.item())

        # register forward hook to the target layer
        module = self.dnn.layer2module(layer)
        handle = module.register_forward_hook(forward_hook)

        return handle

    def synthesize(self, init_image=None, unit=None, lr=0.1, n_iter=30,
                   verbose=True, save_path=None, save_step=None):
        """
        Synthesize the image which maximally activates target layer and channel

        Parameters
        ----------
        init_image : ndarray, Tensor, PIL.Image
            Initialized image.
        unit : tuple
            Set target unit position.
        lr : float
            Learning rate.
        n_iter : int
            The number of iterations
        verbose : bool
            print loss duration iteration or not
        save_path : str
            The directory to save synthesized images.
        save_step : int
            Save out synthesized images for every 'save_step' iterations.
            Only used when save_path is not None.

        Return
        ------
        final_image : ndarray
            The synthesized image with shape as (n_chn, height, width).
        """
        # Hook the selected layer
        handle = self.register_hooks(unit)

        # prepare initialized image
        if init_image is None:
            # Generate a random image
            init_image = torch.rand(3, *self.dnn.img_size, dtype=torch.float32)
        else:
            init_image = ip.to_tensor(init_image).float()
            init_image = copy.deepcopy(init_image)

        self.optimal_image = init_image.unsqueeze(0)
        self.optimal_image.requires_grad_(True)
        self.optimizer = Adam([self.optimal_image], lr=lr)

        # prepare for loss
        self.activ_losses = []
        self.regular_losses = []

        # iteration
        for i in range(n_iter):

            if save_path is not None and save_step is not None:
                if i % save_step == 0:
                    img_out = self.optimal_image[0].detach().numpy().copy()
                    img_out = ip.to_pil(img_out, True)
                    img_out.save(pjoin(save_path, 'synthesized_image_iter{}.jpg'.format(i)))

            # Forward pass layer by layer until the target layer to trigger the hook.
            self.dnn.model(self.optimal_image)

            # computer loss
            loss = self.activ_loss + self.regular_lambda * self.regular_metric()

            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()

            # smooth gradients
            self.smooth_metric(self.factor)

            # Update image
            self.optimizer.step()

            if verbose:
                print('Iteration: {}/{}; Loss: {}'.format(i + 1, n_iter, loss))

            # precondition
            self.precondition_metric(self.GB_radius, lr)

        # trigger hook for the activ_loss of the final synthesized image
        self.dnn.model(self.optimal_image)
        # calculate regular_loss of the final synthesized image
        self.regular_metric()

        # remove hook
        handle.remove()

        # output synthesized image
        final_image = self.optimal_image[0].detach().numpy().copy()
        if save_path is not None:
            final_image = ip.to_pil(final_image, True)
            final_image.save(pjoin(save_path, 'synthesized_image.jpg'))

        return final_image


class MaskedImage(Algorithm):
    """
    Generate masked gray picture for images according to activation changes
    """

    def __init__(self, dnn, layer=None, channel=None, unit=None,
                 stdev_size_thr=1.0, filter_sigma=1.0, target_reduction_ratio=0.9):
        """
        Parameters
        ----------
        dnn : DNN
            A dnnbrain DNN.
        layer : str
            Name of the layer where the algorithm performs on.
        channel : int
            Number of the channel where the algorithm performs on.
        initial_image : ndarray
            Initial image waits for masking.
        unit : tuple
            Position of the target unit.
        """
        super(MaskedImage, self).__init__(dnn, layer, channel)
        self.set_parameters(unit, stdev_size_thr, filter_sigma, target_reduction_ratio)
        self.activ = None
        self.masked_image = None
        self.activ_type = None
        self.row = None
        self.column = None

    def set_parameters(self, unit=None, stdev_size_thr=1.0,
                       filter_sigma=1.0, target_reduction_ratio=0.9):
        """
        Set parameters for mask

        Parameters
        ----------
        unit : tuple
            Position of the target unit.
        stdev_size_thr : float
            Fraction of standard dev threshold for size of blobs, default 1.0.
        filter_sigma : float
            Sigma for final gaussian blur, default 1.0.
        target_reduction_ratio : float
            Reduction ratio to achieve for tightening the mask,default 0.9.
        """
        if isinstance(unit, tuple) and len(unit) == 2:
            self.row, self.column = unit
            self.activ_type = 'unit'
        elif unit == None:
            self.activ_type = 'channel'
        else:
            raise AssertionError('Check unit must be 2-dimentional tuple,like(27,27)')
        self.stdev_size_thr = stdev_size_thr
        self.filter_sigma = filter_sigma
        self.target_reduction_ratio = target_reduction_ratio

    def prepare_test(self, masked_image):
        """
        Transfer pic to tenssor for dnn activation

        Parameters
        ----------
        masked_image : ndarray
            Masked image waits for dnn activation

        returns
        --------
        test_image : tensor
            Pytorch tensor for dnn computation
        """
        test_image = np.repeat(masked_image, 3).reshape((224, 224, 3))
        test_image = test_image.transpose((2, 0, 1))
        test_image = ip.to_tensor(test_image).float()
        test_image = copy.deepcopy(test_image)
        test_image = test_image.unsqueeze(0)
        return test_image

    def register_hooks(self):
        """
        Define register hook and register them to specific layer and channel.
        """
        layer, chn = self.get_layer()

        def forward_hook(module, feat_in, feat_out):
            if self.activ_type == 'channel':
                self.activ = torch.mean(feat_out[0, chn - 1])
            elif self.activ_type == 'unit':
                row = int(self.row)
                column = int(self.column)
                self.activ = feat_out[0, chn - 1, row, column]  # single unit
            self.activ_trace.append(self.activ.item())

        # register forward hook to the target layer
        module = self.dnn.layer2module(layer)
        handle = module.register_forward_hook(forward_hook)

        return handle

    def put_mask(self, initial_image, maxiteration=100):
        """
        Put mask on image

        Parameters
        ----------
        initial_image : ndarray
            Initial image waits for masking.
        maxiteration : int
            The max number of iterations to stop.

        Return
        ------
        masked_image : ndarray
            The masked image with shape as (n_chn, height, width).
        """
        if isinstance(initial_image, np.ndarray):
            if len(initial_image.shape) in [2, 3]:
                img = initial_image
            else:
                raise AssertionError('Check initial_image, only two or three dimentions can be set!')
        else:
            raise AssertionError('Check initial_image to be np.ndarray')
        # define hooks for recording act_loss
        self.activ_trace = []
        handle = self.register_hooks()

        # transpose axis
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = img.transpose((1, 2, 0))

        # degrade dimension
        img = rgb2gray(img)

        # compute the threshold of pixel contrast
        delta = img - img.mean()
        fluc = np.abs(delta)
        thr = np.std(fluc) * self.stdev_size_thr

        # original mask
        mask = convex_hull_image((fluc > thr).astype(float))
        fm = gaussian_filter(mask.astype(float), sigma=self.filter_sigma)
        masked_img = fm * img + (1 - fm) * img.mean()

        # prepare test img and get base acivation
        test_image = self.prepare_test(masked_img)
        self.dnn.model(test_image)
        activation = base_line = self.activ.detach().numpy()

        print('Baseline:', base_line)
        count = 0

        # START
        while (activation > base_line * self.target_reduction_ratio):
            mask = erosion(mask, square(3))

            # print('mask',mask)
            fm = gaussian_filter(mask.astype(float), sigma=self.filter_sigma)
            masked_img = fm * img + (1 - fm) * img.mean()
            test_image = self.prepare_test(masked_img)
            self.dnn.model(test_image)
            activation = - self.activ_loss.detach().numpy()
            print('Activation:', activation)
            count += 1

            if count > maxiteration:
                print('This has been going on for too long! - aborting')
                raise ValueError('The activation does not reduce for the given setting')
                break

        handle.remove()
        masked_image = test_image[0].detach().numpy()
        return masked_image


class MinimalParcelImage(Algorithm):
    """
    A class to generate minimal image for target channels from a DNN model.
    """

    def __init__(self, dnn, layer=None, channel=None, activaiton_criterion='max', search_criterion='max'):
        """
        Parameters
        ----------
        dnn : DNN
            A dnnbrain's DNN object.
        layer : str
            Name of the layer where you focus on.
        channel : int
            Number of the channel where you focus on.
        activaiton_criterion : str
            The criterion of how to pooling activaiton.
        search_criterion : str
            The criterion of how to search minimal image.
        """
        super(MinimalParcelImage, self).__init__(dnn, layer, channel)
        self.set_params(activaiton_criterion, search_criterion)
        self.parcel = None

    def set_params(self, activaiton_criterion='max', search_criterion='max'):
        """
        Set parameter for searching minmal image.

        Parameters
        ----------
        activaiton_criterion : str
            The criterion of how to pooling activaiton, choices=(max, mean, median, L1, L2).
        search_criterion : str
            The criterion of how to search minimal image, choices=(max, fitting curve).
        """
        self.activaiton_criterion = activaiton_criterion
        self.search_criterion = search_criterion

    def _generate_decompose_parcel(self, image, segments):
        """
        Decompose image to multiple parcels using the given segments and
        put each parcel into a separated image with a black background.

        Parameters
        ----------
        image : ndarray
            Shape=(height,width,n_chn).
        segments : ndarray
            Shape (width, height).Integer mask indicating segment labels.

        Return
        ---------
        parcel : ndarray
            Shape (n_parcel,height,width,n_chn).
        """
        self.parcel = np.zeros((np.max(segments) + 1, image.shape[0], image.shape[1], 3), dtype=np.uint8)
        # generate parcel
        for label in np.unique(segments):
            self.parcel[label][segments == label] = image[segments == label]
        return self.parcel

    def felzenszwalb_decompose(self, image, scale=100, sigma=0.5, min_size=50):
        """
        Decompose image to multiple parcels using felzenszwalb method and
        put each parcel into a separated image with a black background.

        Parameters
        ----------
        image : ndarray
            Shape=(height,width,n_chn).

        Return
        ---------
        parcel : ndarray
            Shape=(n_parcel,height,width,n_chn).
        """
        # decompose image
        segments = segmentation.felzenszwalb(image, scale, sigma, min_size)
        # generate parcel
        self.parcel = self._generate_decompose_parcel(image, segments)
        return self.parcel

    def slic_decompose(self, image, n_segments=250, compactness=10, sigma=1):
        """
        Decompose image to multiple parcels using slic method and
        put each parcel into a separated image with a black background.

        Parameters
        ----------
        image : ndarray
            Shape (height,width,n_chn).
        meth : str
            Method to decompose images.

        Return
        ---------
        parcel : ndarray
            Shape=(n_parcel,height,width,n_chn)
        """
        # decompose image
        segments = segmentation.slic(image, n_segments, compactness, sigma)
        # generate parcel
        self.parcel = self._generate_decompose_parcel(image, segments)
        return self.parcel

    def quickshift_decompose(self, image, kernel_size=3, max_dist=6, ratio=0.5):
        """
        Decompose image to multiple parcels using quickshift method and
        put each parcel into a separated image with a black background.

        Parameters
        ----------
        image : ndarray
            Shape (height,width,n_chn).
        meth : str
            Method to decompose images.

        Return
        ---------
        parcel : ndarray
            Shape (n_parcel,height,width,n_chn).
        """
        # decompose image
        segments = segmentation.quickshift(image, kernel_size, max_dist, ratio)
        # generate parcel
        self.parcel = self._generate_decompose_parcel(image, segments)
        return self.parcel

    def sort_parcel(self, order='descending'):
        """
        sort the parcel according the activation of dnn.

        Parameters
        ----------
        order : str
            Ascending or descending.

        Return
        ---------
        parcel : ndarray
            Shape (n_parcel,height,width,n_chn) parcel after sorted.
        """
        # change its shape(n_parcel,n_chn,height,width)
        parcel = self.parcel.transpose((0, 3, 1, 2))
        # compute activation
        dnn_acts = self.dnn.compute_activation(parcel, self.mask).pool(self.activaiton_criterion).get(
            self.mask.layers[0])
        act_all = dnn_acts.flatten()
        # sort the activation in order
        if order == 'descending':
            self.parcel = self.parcel[np.argsort(-act_all)]
        else:
            self.parcel = self.parcel[np.argsort(act_all)]
        return self.parcel

    def combine_parcel(self, indices):
        """
        combine the indexed parcel into a image

        Parameters
        ----------
        indices : list, slice
            Subscript indices.

        Return
        ------
        image_container : ndarray
            Shape=(n_chn,height,width).
        """
        # compose parcel correaspond with indices
        if isinstance(indices, (list, slice)):
            image_compose = np.sum(self.parcel[indices], axis=0)
        else:
            raise AssertionError('Only list and slice indices are supported')
        return image_compose

    def generate_minimal_image(self):
        """
        Generate minimal image. We first sort the parcel by the activiton and
        then iterate to find the combination of the parcels which can maximally
        activate the target channel.

        **Note**: before call this method, you should call xx_decompose method to
        decompose the image into parcels.

        Return
        -------
        image_min : ndarray
            Final minimal images in shape (height,width,n_chn).
        """
        if self.parcel is None:
            raise AssertionError('Please run decompose method to '
                                 'decompose the image into parcels')
        # sort the image
        self.sort_parcel()
        # iterater combine image to get activation
        parcel_add = np.zeros((self.parcel.shape[0], self.parcel.shape[1], self.parcel.shape[2], 3), dtype=np.uint8)
        for index in range(self.parcel.shape[0]):
            parcel_mix = self.combine_parcel(slice(index + 1))
            parcel_add[index] = parcel_mix[np.newaxis, :, :, :]
        # change its shape(n_parcel,n_chn,height,width) to fit dnn_activation
        parcel_add = parcel_add.transpose((0, 3, 1, 2))
        # get activation
        dnn_act = self.dnn.compute_activation(parcel_add, self.mask).pool(self.activaiton_criterion).get(
            self.mask.layers[0])
        act_add = dnn_act.flatten()
        # generate minmal image according to the search_criterion
        intere = 10
        if self.search_criterion == 'max':
            image_min = parcel_add[np.argmax(act_add[0:intere])]
            image_min = np.squeeze(image_min).transpose(1, 2, 0)
        else:
            pass
        return image_min


class MinimalComponentImage(Algorithm):
    """
    A class to generate minmal image for a CNN model using a specific part
    decomposer and optimization criterion.
    """

    def set_params(self, meth='pca', criterion='max'):
        """Set parameter for the estimator"""
        self.meth = meth
        self.criterion = criterion

    def pca_decompose(self):
        pass

    def ica_decompose(self):
        pass

    def sort_componet(self, order='descending'):
        """
        sort the component according the activation of dnn.
        order : str
            Sort order, *'ascending'* or *'descending'*
        """
        pass

    def combine_component(self, index):
        """combine the indexed component into a image"""
        pass

    def generate_minimal_image(self):
        """
        Generate minimal image. We first sort the component by the activiton and
        then iterate to find the combination of the components which can maximally
        activate the target channel.

        Note: before call this method, you should call xx_decompose method to
        decompose the image into parcels.

        Parameters
        ----------
        stim : Stimulus
            Stimulus

        Returns
        ------
        """
        pass


class OccluderDiscrepancyMapping(Algorithm):
    """
    Slide a occluder window on an image, and calculate the change of
    the target channel's activation after each step.
    """

    def __init__(self, dnn, layer=None, channel=None, window=(11, 11), stride=(2, 2), metric='mean'):
        """

        Parameters
        ----------
        dnn : DNN
            A dnnbrain's DNN object.
        layer : str
            Name of the layer that you focus on.
        channel : int
            Number of the channel that you focus on (start with 1).
        window : tuple
            The size of sliding window - (width, height).
        stride : tuple
            The step length of sliding window - (width_step, height_step).
        metric : str
            The metric to summarize the target channel's activation, 'max' or 'mean.
        """
        super(OccluderDiscrepancyMapping, self).__init__(dnn, layer, channel)
        self.set_params(window, stride, metric)

    def set_params(self, window, stride, metric):
        """
        Set parameter for occluder discrepancy mapping.

        Parameters
        ----------
        window : tuple
            The size of sliding window - (width, height).
        stride : tuple
            The step length of sliding window - (width_step, height_step).
        metric : str
            The metric to summarize the target channel's activation, 'max' or 'mean'.
        """
        self.window = window
        self.stride = stride
        self.metric = metric

    def compute(self, image):
        """
        Compute discrepancy map of the image using a occluder window
        moving from top-left to bottom-right.

        Parameters
        ----------
        image : ndarray, Tensor, PIL.Image
            An original image.

        Return
        ---------
        discrepancy_map : ndarray
            Discrepancy activation map.
        """
        # preprocess image
        image = ip.to_array(image)
        image = copy.deepcopy(image)[None, :]

        # initialize discrepancy map
        img_h, img_w = self.dnn.img_size
        win_w, win_h = self.window
        step_w, step_h = self.stride
        n_row = int((img_h - win_h) / step_h + 1)
        n_col = int((img_w - win_w) / step_w + 1)
        discrepancy_map = np.zeros((n_row, n_col))
        activ_ori = array_statistic(self.dnn.compute_activation(image, self.mask).get(self.mask.layers[0]),
                                    self.metric)

        # start computing by moving occluders
        for i in range(n_row):
            start = time.time()
            for j in range(n_col):
                occluded_img = copy.deepcopy(image)
                occluded_img[:, :, step_h * i:step_h * i + win_h, step_w * j:step_w * j + win_w] = 0
                activ_occ = array_statistic(self.dnn.compute_activation(
                    occluded_img, self.mask).get(self.mask.layers[0]), self.metric)
                discrepancy_map[i, j] = activ_ori - activ_occ
            print('Finished: row-{0}/{1}, cost {2} seconds'.format(
                i + 1, n_row, time.time() - start))

        return discrepancy_map


class UpsamplingActivationMapping(Algorithm):
    """
    Resample the target channel's feature map to the input size
    after threshold.
    """

    def __init__(self, dnn, layer=None, channel=None, interp_meth='bicubic', interp_threshold=None):
        """
        Set necessary parameters.

        Parameters
        ----------
        dnn : DNN
            A dnnbrain's DNN object
        layer : str
            Name of the layer where you focus on.
        channel : int
            Number of the channel where you focus on.
        interp_meth : str
            Algorithm used for resampling are
            'nearest', 'bilinear', 'bicubic', 'area'. Default: 'bicubic'.
        interp_threshold : float
            Value is in [0, 1].
            The threshold used to filter the map after resampling.
            For example, if the threshold is 0.58, it means clip the feature map
            with the min as the minimum of the top 42% activation.
        """
        super(UpsamplingActivationMapping, self).__init__(dnn, layer, channel)
        self.set_params(interp_meth, interp_threshold)

    def set_params(self, interp_meth, interp_threshold):
        """
        Set necessary parameters.

        Parameters
        ----------
        interp_meth : str
            Algorithm used for resampling are
            'nearest', 'bilinear', 'bicubic', 'area'. Default: 'bicubic'
        interp_threshold: float
            value is in [0, 1].
            The threshold used to filter the map after resampling.
            For example, if the threshold is 0.58, it means clip the feature map
            with the min as the minimum of the top 42% activation.
        """
        self.interp_meth = interp_meth
        self.interp_threshold = interp_threshold

    def compute(self, image):
        """
        Resample the channel's feature map to input size.

        Parameters
        ---------
        image : ndarray, Tensor, PIL.Image
            An input image.

        Return
        ------
        img_act : ndarray
            Image after resampling, Shape=(height, width).
        """
        # preprocess image
        image = ip.to_array(image)[None, :]

        # compute activation
        img_act = self.dnn.compute_activation(image, self.mask).get(self.mask.layers[0])

        # resample
        img_act = torch.from_numpy(img_act)
        img_act = interpolate(img_act, size=self.dnn.img_size, mode=self.interp_meth)
        img_act = np.squeeze(np.asarray(img_act))

        # threshold
        if self.interp_threshold is not None:
            thr = np.percentile(img_act, self.interp_threshold * 100)
            img_act = np.clip(img_act, thr, None)

        return img_act


class EmpiricalReceptiveField:
    """
    A Class to Estimate Empirical Receptive Field (RF) of a DNN Model.
    """

    def __init__(self, engine=None):
        """
        Parameters
        ----------
        engine : UpsamplingActivationMapping, OccluderDiscrepancyMapping
            The engine to compute empirical receptive field.
        """
        self.set_params(engine)

    def set_params(self, engine):
        """
        Set engine to compute empirical receptive field.

        Parameters
        ----------
        engine : UpsamplingActivationMapping, OccluderDiscrepancyMapping
            Must be an instance of UpsamplingActivationMapping or OccluderDiscrepancyMapping.
        """
        if not isinstance(engine, (UpsamplingActivationMapping, OccluderDiscrepancyMapping)):
            raise TypeError('The engine must be an instance of'
                            'UpsamplingActivationMapping or OccluderDiscrepancyMapping!')
        self.engine = engine

    def generate_rf(self, all_thresed_act):
        """
        Compute RF on Given Image for Target Layer and Channel.

        Parameters
        ----------
        all_thresed_act : ndarray
            Shape must be (n_chn, dnn.img_size).

        Return
        ---------
        empirical_rf_size : np.float64
            Empirical rf size of specific image.
        """
        # init variables
        self.all_thresed_act = all_thresed_act
        sum_act = np.zeros([self.all_thresed_act.shape[0],
                            self.dnn.img_size[0] * 2 - 1, self.dnn.img_size[1] * 2 - 1])
        # compute act of image
        for current_layer in range(self.all_thresed_act.shape[0]):
            cx = int(np.mean(np.where(self.all_thresed_act[current_layer, :, :] ==
                                      np.max(self.all_thresed_act[current_layer, :, :]))[0]))

            cy = int(np.mean(np.where(self.all_thresed_act[current_layer, :, :] ==
                                      np.max(self.all_thresed_act[current_layer, :, :]))[1]))

            sum_act[current_layer,
            self.dnn.img_size[0] - 1 - cx:2 * self.dnn.img_size[0] - 1 - cx,
            self.dnn.img_size[1] - 1 - cy:2 * self.dnn.img_size[1] - 1 - cy] = \
                self.all_thresed_act[current_layer, :, :]

        sum_act = np.sum(sum_act, 0)[int(self.dnn.img_size[0] / 2):int(self.dnn.img_size[0] * 3 / 2),
                  int(self.dnn.img_size[1] / 2):int(self.dnn.img_size[1] * 3 / 2)]
        # get region of receptive field
        plt.imsave('tmp.png', sum_act, cmap='gray')
        rf = cv2.imread('tmp.png', cv2.IMREAD_GRAYSCALE)
        remove('tmp.png')
        rf = cv2.medianBlur(rf, 31)
        _, th = cv2.threshold(rf, self.threshold * 255, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        rf_contour = np.vstack((np.array(contours)[0].squeeze(1), np.array(contours)[1].squeeze(1)))
        empirical_rf_area = 0
        # compute size of rf
        for i in np.unique(rf_contour[:, 0]):
            empirical_rf_area = empirical_rf_area + max(rf_contour[rf_contour[:, 0] == i, 1]) - \
                                min(rf_contour[rf_contour[:, 0] == i, 1])
        empirical_rf_size = np.sqrt(empirical_rf_area)
        return empirical_rf_size

    def compute(self, stimuli, save_path=None):
        """
        Compute empirical receptive field based on input stimulus.

        Parameters
        ----------
        stimuli : Stimulus
            Input stimuli which loaded from files on the disk.
        save_path : str
            Path to save single image's receptive field.
            If None, it will not be saved.

        Return
        ---------
        emp_rf : ndarray
            Mean empirical receptive field of all the input images,
            its shape is equal to the theoretical rf size in specific layer.
        """
        # loaded images
        if not isinstance(stimuli, Stimulus):
            raise TypeError('The input stimuli must be an instance of Stimulus!')
        images = np.zeros((len(stimuli.get('stimID')), 3, 224, 224), dtype=np.uint8)
        for idx, img_id in enumerate(stimuli.get('stimID')):
            image = Image.open(pjoin(stimuli.header['path'], img_id)).convert('RGB')
            image = np.asarray(image).transpose(2, 0, 1)
            image = ip.resize(image, self.engine.dnn.img_size)
            images[idx] = image
        # prepare dnn info
        dnn = self.engine.dnn
        layer = self.engine.mask.layers[0]
        chn = self.engine.mask.get(layer)['chn'][0]
        # prepare rf info
        the_rf = TheoreticalReceptiveField(dnn, layer, chn)
        rf = the_rf.compute()
        layer_int = str(int(dnn.layer2loc[layer][-1]) + 1)
        kernel_size = rf[layer_int]["output_shape"][2:]
        rf_size = rf[layer_int]["r"]
        rf_all = np.zeros((images.shape[0], int(rf_size), int(rf_size)), dtype=np.float32)
        # start computing
        for idx in range(images.shape[0]):
            pic = images[idx]
            # compute upsampling activation map
            img_up = self.engine.compute(pic)
            img_min = np.min(img_up)
            # find the maximum activation in different theoretical rf
            act_all = {}
            patch_all = {}
            range_all = {}
            # loop to compare activations
            for unit_h in range(kernel_size[0]):
                for unit_w in range(kernel_size[1]):
                    rf_standard = np.full((int(rf_size), int(rf_size)), img_min, dtype=np.float32)
                    unit = (unit_h, unit_w)
                    the_rf.set_parameters(unit)
                    rf_range = the_rf.find_region(rf)
                    img_patch = img_up[int(rf_range[0][0]):int(rf_range[0][1]),
                                int(rf_range[1][0]):int(rf_range[1][1])]
                    # enlarge the area if patch size less than rf size
                    if img_patch.shape[0] < rf_size or img_patch.shape[1] < rf_size:
                        rf_standard[0:img_patch.shape[0], 0:img_patch.shape[1]] = img_patch
                    else:
                        rf_standard = img_patch
                    patch_act = np.mean(rf_standard)
                    act_all[unit] = patch_act
                    patch_all[unit] = img_patch
                    range_all[unit] = rf_range
            unit_max = max(act_all, key=act_all.get)
            patch_max = patch_all[unit_max]
            range_max = range_all[unit_max]
            # save single receptive field in the original image
            if not save_path is None:
                img_patch_org = pic[:, int(range_max[0][0]):int(range_max[0][1]),
                                int(range_max[1][0]):int(range_max[1][1])]
                img_patch_org = ip.to_pil(img_patch_org, True)
                img_patch_org.save(pjoin(save_path, f'{idx + 1}.jpg'))
            # integrate all patch
            if int(range_max[0][0]) == 0:
                h_indice = (int(rf_size - patch_max.shape[0]), int(rf_size))
            elif int(range_max[0][1]) == 224:
                h_indice = (0, patch_max.shape[0])
            else:
                h_indice = (0, int(rf_size))
            if int(range_max[1][0]) == 0:
                w_indice = (int(rf_size - patch_max.shape[1]), int(rf_size))
            elif int(range_max[1][1]) == 224:
                w_indice = (0, patch_max.shape[1])
            else:
                w_indice = (0, int(rf_size))
            rf_all[idx][h_indice[0]:h_indice[1],
            w_indice[0]:w_indice[1]] = patch_max
        # compute mean and generate rf
        emp_rf = np.mean(rf_all, axis=0).squeeze()
        return emp_rf


class TheoreticalReceptiveField(Algorithm):
    """
    A Class to Count Theoretical Receptive Field.
    Note: Currently only AlexNet, Vgg16, Vgg19 are supported.
    (All these net are linear structure.)
    """

    def __init__(self, dnn, layer=None, channel=None):
        """
        Parameters
        ----------
        dnn : DNN
            A dnnbrain's DNN object.
        layer : str
            Name of the layer where you focus on.
        channel : int
            Number of the channel where you focus on.
        """
        super(TheoreticalReceptiveField, self).__init__(dnn, layer, channel)

    def set_parameters(self, unit):
        """
        Parameters
        ----------
        unit : tuple
            The unit location in its feature map.
        """
        self.unit = unit

    def compute_size(self):
        if self.dnn.__class__.__name__ == 'AlexNet':
            self.net_struct = {}
            self.net_struct['net'] = [[11, 4, 0], [3, 2, 0], [5, 1, 2], [3, 2, 0],
                                      [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 2, 0]]
            self.net_struct['name'] = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3',
                                       'conv4', 'conv5', 'pool5']

        if self.dnn.__class__.__name__ == 'Vgg11':
            self.net_struct = {}
            self.net_struct['net'] = [[3, 1, 1], [2, 2, 0], [3, 1, 1], [2, 2, 0],
                                      [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1],
                                      [2, 2, 0]]
            self.net_struct['name'] = ['conv1', 'pool1', 'conv2', 'pool2',
                                       'conv3_1', 'conv3_2', 'pool3', 'conv4_1',
                                       'conv4_2', 'pool4', 'conv5_1', 'conv5_2',
                                       'pool5']

        if self.dnn.__class__.__name__ == 'Vgg16':
            self.net_struct['net'] = [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0]]
            self.net_struct['name'] = ['conv1_1', 'conv1_2', 'pool1', 'conv2_1',
                                       'conv2_2', 'pool2', 'conv3_1', 'conv3_2',
                                       'conv3_3', 'pool3', 'conv4_1', 'conv4_2',
                                       'conv4_3', 'pool4', 'conv5_1', 'conv5_2',
                                       'conv5_3', 'pool5']

        if self.dnn.__class__.__name__ == 'Vgg19':
            self.net_struct['net'] = [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1],
                                      [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1],
                                      [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0],
                                      [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                                      [2, 2, 0]]
            self.net_struct['name'] = ['conv1_1', 'conv1_2', 'pool1', 'conv2_1',
                                       'conv2_2', 'pool2', 'conv3_1', 'conv3_2',
                                       'conv3_3', 'conv3_4', 'pool3', 'conv4_1',
                                       'conv4_2', 'conv4_3', 'conv4_4', 'pool4',
                                       'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
                                       'pool5']

        theoretical_rf_size = 1
        # compute size based on net info
        for layer in reversed(range(self.net_struct['name'].index(self.mask.layers[0]) + 1)):
            kernel_size, stride, padding = self.net_struct['net'][layer]
            theoretical_rf_size = ((theoretical_rf_size - 1) * stride) + kernel_size
        return theoretical_rf_size

    def compute(self, batch_size=-1, device="cuda", display=None):
        """
        Compute specific receptive field information for target dnn.
        Only support AlexNet, VGG11!

        Parameters
        ----------
        batch_size : int
            The batch size used in computing.
        device : str
            Input device, please specify 'cuda' or 'cpu'.
        display : bool
            If True, it will show the receptive field information in a table.

        Return
        ---------
        receptive field : OrderedDict
            Receptive field information which contains
            rf_size, feature_map_size, start, jump
        """
        # define params
        model = self.dnn.model
        input_size = (3, *self.dnn.img_size)

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(receptive_field)
                m_key = "%i" % module_idx
                p_key = "%i" % (module_idx - 1)
                receptive_field[m_key] = OrderedDict()
                # define computing formula
                if not receptive_field["0"]["conv_stage"]:
                    print("Enter in deconv_stage")
                    receptive_field[m_key]["j"] = 0
                    receptive_field[m_key]["r"] = 0
                    receptive_field[m_key]["start"] = 0
                else:
                    p_j = receptive_field[p_key]["j"]
                    p_r = receptive_field[p_key]["r"]
                    p_start = receptive_field[p_key]["start"]
                    if class_name == "Conv2d" or class_name == "MaxPool2d":
                        kernel_size = module.kernel_size
                        stride = module.stride
                        padding = module.padding
                        kernel_size, stride, padding = map(self._check_same,
                                                           [kernel_size, stride, padding])
                        receptive_field[m_key]["j"] = p_j * stride
                        receptive_field[m_key]["r"] = p_r + (kernel_size - 1) * p_j
                        receptive_field[m_key]["start"] = p_start + ((kernel_size - 1) / 2 - padding) * p_j
                    elif class_name == "BatchNorm2d" or class_name == "ReLU" or class_name == "Bottleneck":
                        receptive_field[m_key]["j"] = p_j
                        receptive_field[m_key]["r"] = p_r
                        receptive_field[m_key]["start"] = p_start
                    elif class_name == "ConvTranspose2d":
                        receptive_field["0"]["conv_stage"] = False
                        receptive_field[m_key]["j"] = 0
                        receptive_field[m_key]["r"] = 0
                        receptive_field[m_key]["start"] = 0
                    else:
                        raise ValueError("module not ok")
                        pass
                receptive_field[m_key]["input_shape"] = list(input[0].size())
                receptive_field[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    receptive_field[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    receptive_field[m_key]["output_shape"] = list(output.size())
                    receptive_field[m_key]["output_shape"][0] = batch_size

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == model)
            ):
                hooks.append(module.register_forward_hook(hook))

        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"
        # define device in computing
        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        if isinstance(input_size[0], (list, tuple)):
            x = [Variable(torch.rand(2, *in_size)).type(dtype) for in_size in input_size]
        else:
            x = Variable(torch.rand(2, *input_size)).type(dtype)
        # define init params
        receptive_field = OrderedDict()
        receptive_field["0"] = OrderedDict()
        receptive_field["0"]["j"] = 1.0
        receptive_field["0"]["r"] = 1.0
        receptive_field["0"]["start"] = 0.5
        receptive_field["0"]["conv_stage"] = True
        receptive_field["0"]["output_shape"] = list(x.size())
        receptive_field["0"]["output_shape"][0] = batch_size
        # start computing
        hooks = []
        model.features.apply(register_hook)
        model(x)
        for h in hooks:
            h.remove()
        # provide interactive information
        if display == True:
            print(f'Receptive Field Information of {self.dnn.__class__.__name__}'.center(80),
                  "------------------------------------------------------------------------------")
            line_new = "{:>18}  {:>10} {:>12} {:>11} {:>13} ".format("Layer (type)",
                                                                     "map size",
                                                                     "start",
                                                                     "jump",
                                                                     "rf")
            print(line_new)
            print("==============================================================================")
            for layer in receptive_field:
                assert "start" in receptive_field[layer], layer
                assert len(receptive_field[layer]["output_shape"]) == 4
                if layer == '0':
                    layer_out = 'input'
                else:
                    layer_out = list(self.dnn.layer2loc.keys())[
                        [x[-1] for x in self.dnn.layer2loc.values() if x[0] == 'features'].index(str(int(layer) - 1))]
                line_new = "{:5} {:14}  {:>10} {:>10} {:>10} {:>15} ".format(
                    "",
                    layer_out,
                    str(receptive_field[layer]["output_shape"][2:]),
                    str(receptive_field[layer]["start"]),
                    str(receptive_field[layer]["j"]),
                    format(str(receptive_field[layer]["r"]))
                )
                print(line_new)
            print("==============================================================================")
        receptive_field["input_size"] = input_size
        return receptive_field

    def find_region(self, receptive_field):
        """
        Compute specific receptive field range for target dnn, layer and unit.

        Parameters
        ----------
        receptive field : dict
            Receptive field information which contains
            rf_size, feature_map_size, start, jump.

        Return
        --------
        rf_range : list
            The theoretical receptive field region
            example:[(start_h, end_h), (start_w, end_w)].
        """
        layer = str(int(self.dnn.layer2loc[self.mask.layers[0]][-1]) + 1)
        input_shape = receptive_field["input_size"]
        if layer in receptive_field:
            rf_stats = receptive_field[layer]
            assert len(self.unit) == 2
            feat_map_lim = rf_stats['output_shape'][2:]
            if np.any([self.unit[idx] < 0 or
                       self.unit[idx] >= feat_map_lim[idx]
                       for idx in range(2)]):
                raise Exception("Unit position outside spatial extent of the feature tensor")
            rf_range = [(rf_stats['start'] + idx * rf_stats['j'] - rf_stats['r'] / 2,
                         rf_stats['start'] + idx * rf_stats['j'] + rf_stats['r'] / 2)
                        for idx in self.unit]
            if len(input_shape) == 2:
                limit = input_shape
            else:
                limit = input_shape[1:3]
            rf_range = [(max(0, rf_range[axis][0]), min(limit[axis], rf_range[axis][1])) for axis in range(2)]
        else:
            raise KeyError("Layer name incorrect, or not included in the model")
        return rf_range

    def _check_same(self, container):
        """
        Merge elements in the container if they are same.

        Parameters
        ---------
        container : list, tuple
            The containers needed to handle.

        Return
        ---------
        element : int
            Specific elements in the containers.
        """
        if isinstance(container, (list, tuple)):
            assert len(container) == 2 and container[0] == container[1]
            element = container[0]
        else:
            element = container
        return element

class Stimulus:
    """
    Store and handle stimulus-related information
    """

    def __init__(self, header=None, data=None):
        """
        Parameters
        ----------
        header : dict
            Meta-information of stimuli
        data : dict
            Stimulus/behavior data.
            Its values are arrays with shape as (n_stim,).
            It must have the key 'stimID'.
        """
        if header is None:
            self.header = dict()
        else:
            assert isinstance(header, dict), "header must be dict"
            self.header = header

        if data is None:
            self._data = dict()
        else:
            n_stim = len(data['stimID'])
            for v in data.values():
                assert isinstance(v, np.ndarray), "data's value must be an array."
                assert v.shape == (n_stim,), "data's value must be an array with shape as (n_stim,)"
            self._data = data

    def load(self, fname):
        """
        Load stimulus-related information

        Parameters
        ----------
        fname : str
            File name with suffix as .stim.csv
        """
        stim_file = fio.StimulusFile(fname)
        stimuli = stim_file.read()
        self._data = stimuli.pop('data')
        self.header = stimuli

    def save(self, fname):
        """
        Save stimulus-related information

        Parameters
        ----------
        fname : str
            File name with suffix as .stim.csv
        """
        stim_file = fio.StimulusFile(fname)
        header = self.header.copy()
        stim_file.write(header.pop('type'), header.pop('path'),
                        self._data, **header)

    def get(self, item):
        """
        Get a column of data according to the item

        Parameters
        ----------
        item : str
            Item name of each column

        Returns
        -------
        col : array
            A column of data
        """
        return self._data[item]

    def set(self, item, value):
        """
        Set a column of data according to the item

        Parameters
        ----------
        item : str
            Item name of the column
        value : array_like
            An array_like data with shape as (n_stim,)
        """
        self._data[item] = np.asarray(value)

    def delete(self, item):
        """
        Delete a column of data according to item

        Parameters
        ----------
        item : str
            Item name of each column
        """
        self._data.pop(item)

    def permutate(self, item):
        """
        Permutate a column of data according to the item

        Parameters
        ----------
        item : str
            Item name of the column

        Returns
        -------
        stim : Stimulus
            A copy of the current instance of Stimulus
            except for the permutation on the specified column.
        """
        indices = list(range(len(self._data['stimID'])))
        random.shuffle(indices)

        stim = Stimulus()
        stim.header = deepcopy(self.header)
        stim.set(item, self.get(item)[indices])
        items = self.items
        items.remove(item)
        for item in items:
            stim.set(item, deepcopy(self.get(item)))

        return stim

    @property
    def items(self):
        """
        Get list of items

        Returns
        -------
        items : list
            The list of items
        """
        return list(self._data.keys())

    def __len__(self):
        """
        the length of the Stimulus object

        Returns
        -------
        length : int
            The number of stimulus IDs
        """
        return len(self._data['stimID'])

    def __getitem__(self, indices):
        """
        Get part of the Stimulus object by imitating 2D array's subscript index

        Parameters
        ----------
        indices : int,list,tuple,slice
            Subscript indices

        Returns
        -------
        stim : Stimulus
            A part of the self.
        """
        # parse subscript indices
        if isinstance(indices, int):
            # regard it as row index
            # get all columns
            rows = [indices]
            cols = self.items
        elif isinstance(indices, (slice, list)):
            # regard it all as row indices
            # get all columns
            rows = indices
            cols = self.items
        elif isinstance(indices, tuple):
            if len(indices) == 0:
                # get all rows and columns
                rows = slice(None, None, None)
                cols = self.items
            elif len(indices) == 1:
                # regard the only element as row indices
                # get all columns
                if isinstance(indices[0], int):
                    # regard it as row index
                    rows = [indices[0]]
                elif isinstance(indices[0], (slice, list)):
                    # regard it all as row indices
                    rows = indices[0]
                else:
                    raise IndexError("only integer, slices (`:`), list are valid row indices")
                cols = self.items
            elif len(indices) == 2:
                # regard the first element as row indices
                # regard the second element as column indices
                rows, cols = indices
                if isinstance(rows, int):
                    # regard it as row index
                    rows = [rows]
                elif isinstance(rows, (slice, list)):
                    # regard it all as row indices
                    pass
                else:
                    raise IndexError("only integer, slices (`:`), list are valid row indices")

                if isinstance(cols, int):
                    # get a column according to an integer
                    cols = [self.items[cols]]
                elif isinstance(cols, str):
                    # get a column according to an string
                    cols = [cols]
                elif isinstance(cols, list):
                    if np.all([isinstance(i, int) for i in cols]):
                        # get columns according to a list of integers
                        cols = [self.items[i] for i in cols]
                    elif np.all([isinstance(i, str) for i in cols]):
                        # get columns according to a list of strings
                        pass
                    else:
                        raise IndexError("only integer [list], string [list] and slices (`:`) "
                                         "are valid column indices")
                elif isinstance(cols, slice):
                    # get columns according to a slice
                    cols = self.items[cols]
                else:
                    raise IndexError("only integer [list], string [list] and slices (`:`) "
                                     "are valid column indices")
            else:
                raise IndexError("This is a 2D data, "
                                 "and can't support more than 3 subscript indices!")
        else:
            raise IndexError("only integer, slices (`:`), list and tuple are valid indices")

        # get part of self
        stim = Stimulus()
        stim.header = deepcopy(self.header)
        for item in cols:
            stim.set(item, self.get(item)[rows])

        return stim


class Activation:
    """
    DNN activation
    """

    def __init__(self, layer=None, value=None):
        """
        Parameters
        ----------
        layer : str
            Layer name
        value : array
            4D DNN activation array with shape (n_stim, n_chn, n_row, n_col).
            It will be ignored if layer is None.
        """
        if layer is None:
            self._activation = dict()
        else:
            assert value is not None, "value can't be None if layer is not None."
            self.set(layer, value)

    def load(self, fname, dmask=None):
        """
        Load DNN activation

        Parameters
        ----------
        fname : str
            DNN activation file
        dmask : Mask
            The mask includes layers/channels/rows/columns of interest.
        """
        if dmask is not None:
            dmask_dict = dict()
            for layer in dmask.layers:
                dmask_dict[layer] = dmask.get(layer)
        else:
            dmask_dict = None

        self._activation = fio.ActivationFile(fname).read(dmask_dict)

    def save(self, fname):
        """
        Save DNN activation

        Parameters
        ----------
        fname : str
            Output file of DNN activation
        """
        fio.ActivationFile(fname).write(self._activation)

    def get(self, layer):
        """
        Get DNN activation

        Parameters
        ----------
        layer : str
            Layer name

        Returns
        -------
        act_layer : array
            (n_stim, n_chn, n_row, n_col) array
        """
        return self._activation[layer]

    def set(self, layer, value):
        """
        Set DNN activation

        Parameters
        ----------
        layer : str
            Layer name
        value : array
            4D DNN activation array with shape (n_stim, n_chn, n_row, n_col)
        """
        self._activation[layer] = value

    def delete(self, layer):
        """
        Delete DNN activation

        Parameters
        ----------
        layer : str
            Layer name
        """
        self._activation.pop(layer)

    def concatenate(self, activations):
        """
        Concatenate activations from different batches of stimuli

        Parameters
        ----------
        activations : list
            A list of Activation objects

        Returns
        -------
        activation : Activation
            DNN activation
        """
        # check availability
        for i, v in enumerate(activations, 1):
            if not isinstance(v, Activation):
                raise TypeError('All elements in activations must be instances of Activation!')
            if sorted(self.layers) != sorted(v.layers):
                raise ValueError("The element{}'s layers mismatch with self!".format(i))

        # concatenate
        activation = Activation()
        for layer in self.layers:
            # concatenate activation
            data = [v.get(layer) for v in activations]
            data.insert(0, self.get(layer))
            data = np.concatenate(data)
            activation.set(layer, data)

        return activation

    @property
    def layers(self):
        """
        Get all layers in the Activation

        Returns
        -------
        layers : list
           The list of layers.
        """
        return list(self._activation.keys())

    def mask(self, dmask):
        """
        Mask DNN activation

        Parameters
        ----------
        dmask : Mask
            The mask includes layers/channels/rows/columns of interest.

        Returns
        -------
        activation : Activation
            DNN activation
        """
        activation = Activation()
        for layer in dmask.layers:
            mask = dmask.get(layer)
            data = dnn_mask(self.get(layer), mask.get('chn'),
                            mask.get('row'), mask.get('col'))
            activation.set(layer, data)

        return activation

    def pool(self, method):
        """
        Pooling DNN activation for each channel

        Parameters
        ----------
        method : str
            Pooling method, choices=(max, mean, median, L1, L2)

        Returns
        -------
        activation : Activation
            DNN activation
        """
        activation = Activation()
        for layer, data in self._activation.items():
            data = array_statistic(data, method, (2, 3), True)
            activation.set(layer, data)

        return activation

    def fe(self, method, n_feat, axis=None):
        """
        Extract features of DNN activation

        Parameters
        ----------
        method : str
            Feature extraction method, choices are as follows:

            +-------------+---------------------------------------------+
            | Method name |              Model description              |
            +=============+=============================================+
            |     pca     | use n_feat principal components as features |
            +-------------+---------------------------------------------+
            |    hist     | use histogram of activation as features     |
            |             | Note: n_feat equal-width bins in the        |
            |             | given range will be used!                   |
            +-------------+---------------------------------------------+
            |     psd     | use power spectral density as features      |
            +-------------+---------------------------------------------+
        n_feat : int, float
            The number of features to extract.
            Note: It can be a float only when the method is pca.
        axis : str
            axis for feature extraction, choices=(chn, row_col)

        Returns
        -------
        activation : Activation
            DNN activation
        """
        activation = Activation()
        for layer, data in self._activation.items():
            data = dnn_fe(data, method, n_feat, axis)
            activation.set(layer, data)

        return activation

    def convolve_hrf(self, onsets, durations, n_vol, tr, ops=100):
        """
        Convolve DNN activation with HRF and align with the timeline of BOLD signal

        Parameters
        ----------
        onsets : array_like
            In sec. size = n_event
        durations : array_like
            In sec. size = n_event
        n_vol : int
            The number of volumes of BOLD signal
        tr : float
            Repeat time in second
        ops : int
            Oversampling number per second

        Returns
        -------
        activation : Activation
            DNN activation
        """
        activation = Activation()
        for layer, data in self._activation.items():
            n_stim, n_chn, n_row, n_col = data.shape
            data = convolve_hrf(data.reshape(n_stim, -1), onsets, durations,
                                n_vol, tr, ops)
            data = data.reshape(n_vol, n_chn, n_row, n_col)
            activation.set(layer, data)

        return activation

    def _check_arithmetic(self, other):
        """
        Check availability of the arithmetic operation for self

        Parameters
        ----------
        other : Activation
            DNN activation
        """
        if not isinstance(other, Activation):
            raise TypeError("unsupported operand type(s): "
                            "'{0}' and '{1}'".format(type(self), type(other)))
        assert sorted(self.layers) == sorted(other.layers), \
            "The two object's layers mismatch!"
        for layer in self.layers:
            assert self.get(layer).shape == other.get(layer).shape, \
                "{}'s activation shape mismatch!".format(layer)

    def __add__(self, other):
        """
        Define addition operation

        Parameters
        ----------
        other : Activation
            DNN activation

        Returns
        -------
        activation : Activation
            DNN activation
        """
        self._check_arithmetic(other)

        activation = Activation()
        for layer in self.layers:
            data = self.get(layer) + other.get(layer)
            activation.set(layer, data)

        return activation

    def __sub__(self, other):
        """
        Define subtraction operation

        Parameters
        ----------
        other : Activation
            DNN activation

        Returns
        -------
        activation : Activation
            DNN activation
        """
        self._check_arithmetic(other)

        activation = Activation()
        for layer in self.layers:
            data = self.get(layer) - other.get(layer)
            activation.set(layer, data)

        return activation

    def __mul__(self, other):
        """
        Define multiplication operation

        Parameters
        ----------
        other : Activation
            DNN activation

        Returns
        -------
        activation : Activation
            DNN activation
        """
        self._check_arithmetic(other)

        activation = Activation()
        for layer in self.layers:
            data = self.get(layer) * other.get(layer)
            activation.set(layer, data)

        return activation

    def __truediv__(self, other):
        """
        Define true division operation

        Parameters
        ----------
        other : Activation
            DNN activation

        Returns
        -------
        activation : Activation
            DNN activation
        """
        self._check_arithmetic(other)

        activation = Activation()
        for layer in self.layers:
            data = self.get(layer) / other.get(layer)
            activation.set(layer, data)

        return activation

    def __getitem__(self, indices):
        """
        Get part of Activation along stimulus axis

        Parameters
        ----------
        indices : int, list, slice
            indices of stimulus axis

        Returns
        -------
        activation : Activation
            DNN activation
        """
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, (list, slice)):
            pass
        else:
            raise IndexError("only integer, slices (`:`), and list are valid indices")

        activation = Activation()
        for layer in self.layers:
            data = self.get(layer)[indices]
            activation.set(layer, data)

        return activation


class Mask:
    """
    DNN mask
    """

    def __init__(self, layer=None, channels='all', rows='all', columns='all'):
        """
        Parameters
        ----------
        layer : str
            Layer name.
            If layer is None, other parameters will be ignored.
        channels : str, list
            Channels of interest.
            If is str, it must be 'all' which means all channels.
            If is list, its elements are serial numbers of channels.
        rows : str, list
            Rows of interest.
            If is str, it must be 'all' which means all rows.
            If is list, its elements are serial numbers of rows.
        columns : str, list
            Columns of interest.
            If is str, it must be 'all' which means all columns.
            If is list, its elements are serial numbers of columns.
        """
        self._dmask = dict()
        if layer is not None:
            self.set(layer, channels=channels, rows=rows, columns=columns)

    def load(self, fname):
        """
        Load DNN mask, the whole mask will be overrode.

        Parameters
        ----------
        fname : str
            DNN mask file
        """
        self._dmask = fio.MaskFile(fname).read()

    def save(self, fname):
        """
        Save DNN mask

        Parameters
        ----------
        fname : str
            Output file name of DNN mask
        """
        fio.MaskFile(fname).write(self._dmask)

    def get(self, layer):
        """
        Get mask of a layer

        Parameters
        ---------
        layer : str
            Layer name

        Returns
        -------
        mask : dict
            The mask of a specific layer
        """
        return self._dmask[layer]

    def set(self, layer, **kwargs):
        """
        Set DNN mask

        Parameters
        ----------
        layer : str
            Layer name.
            If layer is new, its corresponding mask value will be initialized as 'all'.
        kwargs : dict
            Keyword arguments.
            Only three keywords ('channels', 'rows', 'columns') are valid.

            +-------------+-------------+----------------------------------------------+
            |   Keywords  |    Option   |                Description                   |
            +=============+=============+==============================================+
            |   channels  |     str     | It must be 'all' which means all channels.   |
            |             +-------------+----------------------------------------------+
            |             |    list     | Its elements are serial numbers of channels. |
            +-------------+-------------+----------------------------------------------+
            |     rows    |     str     | It must be 'all' which means all rows.       |
            |             +-------------+----------------------------------------------+
            |             |    list     | Its elements are serial numbers of rows.     |
            +-------------+-------------+----------------------------------------------+
            |   columns   |     str     | It must be 'all' which means all columns.    |
            |             +-------------+----------------------------------------------+
            |             |    list     | Its elements are serial numbers of columns.  |
            +-------------+-------------+----------------------------------------------+
        """
        # assertion
        for k, v in kwargs.items():
            assert k in ('channels', 'rows', 'columns'), \
                "keyword must be one of ('channels', 'rows', 'columns')"
            assert v == 'all' or isinstance(v, list), \
                f"{k} must be 'all' or list of non-negative integers"

        if layer not in self._dmask:
            self._dmask[layer] = {'chn': 'all', 'row': 'all', 'col': 'all'}
        if 'channels' in kwargs:
            self._dmask[layer]['chn'] = kwargs['channels']
        if 'rows' in kwargs:
            self._dmask[layer]['row'] = kwargs['rows']
        if 'columns' in kwargs:
            self._dmask[layer]['col'] = kwargs['columns']

    def copy(self):
        """
        Make a copy of the DNN mask

        Returns
        -------
        dmask : Mask
            The mask includes layers/channels/rows/columns of interest.
        """
        dmask = Mask()
        dmask._dmask = deepcopy(self._dmask)

        return dmask

    def delete(self, layer):
        """
        Delete a layer

        Parameters
        ---------
        Layer : str
            Layer name
        """
        self._dmask.pop(layer)

    def clear(self):
        """
        Empty the DNN mask
        """
        self._dmask.clear()

    @property
    def layers(self):
        """
        Get all layers in the Mask

        Returns
        -------
        layers : list
           The list of layers.
        """
        return list(self._dmask.keys())


class RDM:
    """
    Representation distance matrix
    """

    def __init__(self):
        self.rdm_type = None
        self._rdm_dict = dict()

    def load(self, fname):
        """
        load RDM

        Parameters
        ----------
        fname : str
            File name with suffix as .rdm.h5
        """
        self.rdm_type, self._rdm_dict = fio.RdmFile(fname).read()

    def save(self, fname):
        """
        Save RDM

        Parameters
        ----------
        fname : str
            File name with suffix as .rdm.h5
        """
        fio.RdmFile(fname).write(self.rdm_type, self._rdm_dict)

    def get(self, key, triu=False):
        """
        Get RDM according its key.

        Parameters
        ----------
        key : str
            The key of the RDM
        triu : bool
            If True, get RDM as the upper triangle vector.
            If False, get RDM as the square matrix.

        Returns
        -------
        rdm_arr : ndarray
            RDM

            If rdm_type is bRDM:
            Its shape is ((n_item^2-n_item)/2,) or (n_item, n_item).

            If rdm_type is dRDM:
            Its shape is (n_iter, (n_item^2-n_item)/2) or (n_iter, n_item, n_item).
        """
        rdm_arr = self._rdm_dict[key]
        if not triu:
            idx_arr = np.tri(self.n_item, k=-1, dtype=np.bool_).T
            if self.rdm_type == 'bRDM':
                rdm_tmp = np.zeros((self.n_item, self.n_item))
                rdm_tmp[idx_arr] = rdm_arr
            elif self.rdm_type == 'dRDM':
                rdm_tmp = np.zeros((rdm_arr.shape[0], self.n_item, self.n_item))
                rdm_tmp[:, idx_arr] = rdm_arr
            else:
                raise TypeError("Set rdm_type to bRDM or dRDM at first!")
            rdm_arr = rdm_tmp

        return rdm_arr

    def set(self, key, rdm_arr, triu=False):
        """
        Set RDM according its key.

        Parameters
        ----------
        key : str
            The key of the RDM
        rdm_arr : ndarray
            RDM

            If rdm_type is bRDM:
            Its shape is ((n_item^2-n_item)/2,) or (n_item, n_item).

            If rdm_type is dRDM:
            Its shape is (n_iter, (n_item^2-n_item)/2) or (n_iter, n_item, n_item).
        triu : bool
            If True, RDM will be regarded as the upper triangle vector.
            If False, RDM will be regarded as the square matrix.
        """
        if self.rdm_type == 'bRDM':
            if triu:
                assert rdm_arr.ndim == 1, \
                    "If triu is True, bRDM's shape must be ((n_item^2-n_item)/2,)."
                self._rdm_dict[key] = rdm_arr
            else:
                assert rdm_arr.ndim == 2 and rdm_arr.shape[0] == rdm_arr.shape[1], \
                    "If triu is False, bRDM's shape must be (n_item, n_item)."
                self._rdm_dict[key] = rdm_arr[np.tri(rdm_arr.shape[0], k=-1, dtype=np.bool_).T]
        elif self.rdm_type == 'dRDM':
            if triu:
                assert rdm_arr.ndim == 2, \
                    "If triu is True, dRDM's shape must be (n_iter, (n_item^2-n_item)/2)."
                self._rdm_dict[key] = rdm_arr
            else:
                assert rdm_arr.ndim == 3 and rdm_arr.shape[1] == rdm_arr.shape[2], \
                    "If triu is False, dRDM's shape must be (n_iter, n_item, n_item)."
                self._rdm_dict[key] = rdm_arr[:, np.tri(rdm_arr.shape[1], k=-1, dtype=np.bool_).T]
        else:
            raise TypeError("Set rdm_type to bRDM or dRDM at first!")

    @property
    def keys(self):
        """
        Get keys of RDM dictionary

        Returns
        -------
        keys : list
            The list of keys
        """
        if self._rdm_dict:
            keys = list(self._rdm_dict.keys())
        else:
            raise ValueError("The RDM dictionary is empty.")

        return keys

    @property
    def n_item(self):
        """
        Get the number of items of RDM

        Returns
        -------
        n_item : int
            The number of items
        """
        k = self.keys[0]
        if self.rdm_type == 'bRDM':
            n = self._rdm_dict[k].shape[0]
        elif self.rdm_type == 'dRDM':
            n = self._rdm_dict[k].shape[1]
        else:
            raise TypeError("Set rdm_type to bRDM or dRDM at first!")
        n_item = int((1 + np.sqrt(1 + 8 * n)) / 2)

        return n_item


class DnnProbe:
    """
    Decode DNN activation to behavior data. As a result, |br|
    probe the ability of DNN activation to predict the behavior.
    """

    def __init__(self, dnn_activ=None, map_type=None, estimator=None,
                 cv=5, scoring=None):
        """
        Parameters
        ----------
        dnn_activ : Activation
            DNN activation
        map_type : str
            choices=(uv, mv)
            uv: univariate mapping
            mv: multivariate mapping
        estimator : str | sklearn estimator or pipeline
            If is str, it is a name of a estimator used to do mapping.
            If is 'corr', it just uses correlation rather than prediction.
                And the map_type must be 'uv'.
        cv : int
            the number of cross validation folds.
        scoring : str or callable
            the method to evaluate the predictions on the test set.
        """
        self.set_activ(dnn_activ)
        self.set_mapper(map_type, estimator, cv, scoring)

    def set_activ(self, dnn_activ):
        """
        Set DNN activation

        Parameters
        ----------
        dnn_activ : Activation
            DNN activation
        """
        self.dnn_activ = dnn_activ

    def set_mapper(self, map_type, estimator, cv, scoring):
        """
        Set mapping attributes

        Parameters
        ----------
        map_type : str
            choices=(uv, mv) |br|
            uv: univariate mapping |br|
            mv: multivariate mapping
        estimator : str | sklearn estimator or pipeline
            Estimator used in mapping. |br|
            If is str, it is a name of a estimator used to do mapping. |br|
            If the name is 'corr', it just uses correlation rather than prediction, |br|
            and the map_type must be 'uv'.
        cv : int
            The number of cross validation folds.
        scoring : str or callable
            The method to evaluate the predictions on the test set.
        """
        if map_type is None:
            return
        elif map_type == 'uv':
            self.mapper = UnivariateMapping(estimator, cv, scoring)
        elif map_type == 'mv':
            self.mapper = MultivariateMapping(estimator, cv, scoring)
        else:
            raise ValueError('map_type must be one of the (uv, mv).')

    def probe(self, beh_data, iter_axis=None):
        """
        Probe the ability of DNN activation to predict the behavior.

        Parameters
        ----------
        beh_data : ndarray
            Behavior data with shape as (n_stim, n_beh)
        iter_axis : str
            Iterate along the specified axis. Different map type have different operations.

            +-------+---------+----------------------------------------------------------+
            | map   |iter_axis|  description                                             |
            | type  |         |                                                          |
            +=======+=========+==========================================================+
            | uv    | channel |Summarize the maximal prediction score for each channel   |
            |       +---------+----------------------------------------------------------+
            |       | row_col |Summarize the maximal prediction score for each position  |
            |       |         |(row_idx, col_idx)                                        |
            |       +---------+----------------------------------------------------------+
            |       | None    |Summarize the maximal prediction score for the whole layer|
            +-------+---------+----------------------------------------------------------+
            |  mv   | channel |Multivariate prediction using all units in each channel   |
            |       +---------+----------------------------------------------------------+
            |       | row_col |Multivariate prediction using all units in each           |
            |       |         |position (row_idx, col_idx)                               |
            |       +---------+----------------------------------------------------------+
            |       | None    |Multivariate prediction using all units in the whole layer|
            +-------+---------+----------------------------------------------------------+

        Returns
        -------
        probe_dict : dict
            A dict containing the score information

            +-------+---------+-----------------------------------------------------------------------+
            |       |         |                           First value                                 |
            |       |         +-----------+-----------------------------------------------------------+
            | Map   |First    |Second     |                       Second value                        |
            | type  |key      |key        |                                                           |
            +=======+=========+===========+===========================================================+
            |  uv   | layer   | score     |If estimator type is correlation, it's an                  |
            |       |         |           |array with shape as (n_iter, n_beh). |br|                  |
            |       |         |           |Each element is the maximal pearson r among all            |
            |       |         |           |features at corresponding iteration correlating            |
            |       |         |           |to the corresponding behavior. |br|                        |
            |       |         |           |If estimator type is regressor or classifier,              |
            |       |         |           |it's an array with shape as (n_iter, n_beh, cv). |br|      |
            |       |         |           |For each iteration and behavior, the third axis            |
            |       |         |           |contains scores of each cross validation fold,             |
            |       |         |           |when using the feature with maximal score                  |
            |       |         |           |to predict the corresponding behavior.                     |
            |       | (str)   +-----------+-----------------------------------------------------------+
            |       |         | location  |An array with shape as (n_iter, n_beh, 3) |br|             |
            |       |         |           |Max locations of the max scores, the |br|                  |
            |       |         |           |size 3 of the third dimension means |br|                   |
            |       |         |           |channel, row and column respectively.                      |
            |       |         +-----------+-----------------------------------------------------------+
            |       |         | model     |An array with shape as (n_iter, n_beh). |br|               |
            |       |         |           |Fitted models of the max scores. |br|                      |
            |       |         |           |Note: not exists when estimator type is correlation.       |
            |       |         +-----------+-----------------------------------------------------------+
            |       |         | conf_m    |An array with shape as (n_iter, n_beh, cv) |br|            |
            |       |         |           |The third dimension means confusion matrices               |
            |       |         |           |(n_label, n_label) of each cross validation                |
            |       |         |           |fold of the max scores. |br|                               |
            |       |         |           |Note: only exists when estimator type is classifier.       |
            +-------+---------+-----------+-----------------------------------------------------------+
            |  mv   | layer   | score     |An array with shape as (n_iter, n_beh, cv) |br|            |
            |       |         |           |The third dimension means scores of each                   |
            |       | (str)   |           |cross validation fold at each iteration                    |
            |       |         |           |and behavior.                                              |
            |       |         +-----------+-----------------------------------------------------------+
            |       |         | model     |An array with shape as (n_iter, n_beh). |br|               |
            |       |         |           |Each element is a model fitted at the |br|                 |
            |       |         |           |corresponding iteration and behavior.                      |
            |       |         +-----------+-----------------------------------------------------------+
            |       |         | conf_m    |An array with shape as (n_iter, n_beh, cv). |br|           |
            |       |         |           |The third dimension means confusion matrices               |
            |       |         |           |(n_label, n_label) of each cross validation                |
            |       |         |           |fold at the corresponding iteration and behavior. |br|     |
            |       |         |           |Note: only exists when estimator type is classifier.       |
            +-------+---------+-----------+-----------------------------------------------------------+

            .. |br| raw:: html

               <br/>
        """
        _, n_beh = beh_data.shape

        probe_dict = dict()
        for layer in self.dnn_activ.layers:
            # get DNN activation and reshape it to 3D
            activ = self.dnn_activ.get(layer)
            n_stim, n_chn, n_row, n_col = activ.shape
            n_row_col = n_row * n_col
            activ = activ.reshape((n_stim, n_chn, n_row_col))

            # transpose axis to make activ's shape as (n_stimulus, n_iterator, n_element)
            if iter_axis is None:
                activ = activ.reshape((n_stim, 1, -1))
            elif iter_axis == 'row_col':
                activ = activ.transpose((0, 2, 1))
            elif iter_axis == 'channel':
                pass
            else:
                raise ValueError("Unsupported iter_axis:", iter_axis)
            n_stim, n_iter, n_elem = activ.shape

            # prepare layer dict
            if self.mapper.estimator_type == 'correlation':
                probe_dict[layer] = {'score': np.zeros((n_iter, n_beh))}
            elif self.mapper.estimator_type == 'regressor':
                probe_dict[layer] = {
                    'score': np.zeros((n_iter, n_beh, self.mapper.cv)),
                    'model': np.zeros((n_iter, n_beh), dtype=np.object)
                }
            else:
                probe_dict[layer] = {
                    'score': np.zeros((n_iter, n_beh, self.mapper.cv)),
                    'model': np.zeros((n_iter, n_beh), dtype=np.object),
                    'conf_m': np.zeros((n_iter, n_beh, self.mapper.cv), dtype=np.object)
                }

            # start probing
            if isinstance(self.mapper, UnivariateMapping):
                probe_dict[layer]['location'] = np.zeros((n_iter, n_beh, 3), dtype=np.int)

                # start iteration
                for iter_idx in range(n_iter):
                    data = self.mapper.map(activ[:, iter_idx, :], beh_data)
                    for k, v in data.items():
                        if k == 'location':
                            if iter_axis is None:
                                chn_idx = v // n_row_col
                                row_idx = v % n_row_col // n_col
                                col_idx = v % n_row_col % n_col
                            elif iter_axis == 'channel':
                                chn_idx = iter_idx
                                row_idx = v // n_col
                                col_idx = v % n_col
                            else:
                                chn_idx = v
                                row_idx = iter_idx // n_col
                                col_idx = iter_idx % n_col
                            probe_dict[layer][k][iter_idx, :, 0] = chn_idx + 1
                            probe_dict[layer][k][iter_idx, :, 1] = row_idx + 1
                            probe_dict[layer][k][iter_idx, :, 2] = col_idx + 1
                        else:
                            probe_dict[layer][k][iter_idx] = v
                    print('Layer-{} iter-{}/{}'.format(layer, iter_idx + 1, n_iter))
            else:
                # start iteration
                for iter_idx in range(n_iter):
                    data = self.mapper.map(activ[:, iter_idx, :], beh_data)
                    for k, v in data.items():
                        probe_dict[layer][k][iter_idx] = v

                    print('Layer-{} iter-{}/{}'.format(layer, iter_idx + 1, n_iter))

        return probe_dict