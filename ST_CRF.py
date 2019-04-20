from multiprocessing import Pool
import os
import numpy as np
from skimage.transform import resize
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

def crf(sm_mask_one, img_one, num_class, input_size, mask_size, num_iter):
    sm_u = np.transpose(resize(np.transpose(sm_mask_one, [1,2,0]), input_size, mode='constant'),[2,0,1])

    U = unary_from_softmax(sm_u)

    d = dcrf.DenseCRF2D(input_size[0], input_size[1], num_class)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    # d.addPairwiseBilateral(sxy=(30,30), srgb=(13,13,13), rgbim=img_one.astype(np.uint8), compat=20, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=img_one.astype(np.uint8), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(num_iter)
    return np.array(Q).reshape((num_class, input_size[0], input_size[1]))


class STCRFLayer():
    def __init__(self, flag_multi_process = False):
        self.num_class = 21
        self.min_prob = 0.0001
        self.mask_size = [41, 41]
        self.input_size = [321, 321]
        self.num_iter = 3
        self.flag_multi_process = flag_multi_process
        if flag_multi_process:
            num_cores = os.cpu_count()
            self.pool = Pool(processes=num_cores)

    def run(self, sm_mask, img):
        self.mask_size = [sm_mask.shape[-2], sm_mask.shape[-1]]
        self.input_size = [img.shape[-3], img.shape[-2]]

        if self.flag_multi_process:
            return self.run_parallel(sm_mask, img)
        else:
            return self.run_single(sm_mask, img)

    def run_single(self, sm_mask, img): # the input array are detached numpy already
        batch_size = sm_mask.shape[0]
        result_big = np.zeros((sm_mask.shape[0], sm_mask.shape[1], self.input_size[0], self.input_size[1]))
        result_small = np.zeros(sm_mask.shape)
        for i in range(batch_size):

            sm_u = np.transpose(resize(np.transpose(sm_mask[i], [1,2,0]), self.input_size, mode='constant'),[2,0,1])

            U = unary_from_softmax(sm_u)

            d = dcrf.DenseCRF2D(self.input_size[1], self.input_size[0], self.num_class)
            d.setUnaryEnergy(U)

            d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
            # d.addPairwiseBilateral(sxy=(30,30), srgb=(13,13,13), rgbim=img[i].astype(np.uint8), compat=20, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
            d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=img[i].astype(np.uint8), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

            Q = d.inference(self.num_iter)
            result_big[i] = np.array(Q).reshape((self.num_class, self.input_size[0], self.input_size[1]))
            result_small[i] = np.transpose(resize(np.transpose(result_big[i],[1,2,0]), self.mask_size, mode='constant'), [2,0,1])

        return result_big, result_small

    def run_parallel(self, sm_mask, img): # flag_train is for the strange dif between train & test in org SEC
        batch_size = sm_mask.shape[0]
        result_big = np.zeros((sm_mask.shape[0], sm_mask.shape[1], self.input_size[0], self.input_size[1]))
        result_small = np.zeros(sm_mask.shape)

        # temp = self.pool.starmap(self.crf,[(sm_mask[i], img[i]) for i in range(batch_size)])
        temp = self.pool.starmap(crf,[(sm_mask[i], img[i], self.num_class, self.input_size, self.mask_size, self.num_iter) for i in range(batch_size)])
        for i in range(batch_size):
            result_big[i] = temp[i]
            result_small[i] = np.transpose(resize(np.transpose(result_big[i],[1,2,0]), self.mask_size, mode='constant'), [2,0,1])

        return result_big, result_small
