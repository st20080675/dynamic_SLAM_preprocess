import torch
import numpy as np
import torch.nn as nn

import datetime
import os
import logging
import sys


def model_loader(args):

    if args.model == 'resnet38':
        import network.resnet38_cls as resnet38_cls
        net = resnet38_cls.Net()
    elif args.model == "resnet50":
        import network.resnet50 as resnet50
        net = resnet50.Net(num_classes=args.num_classes)
    elif args.model == 'vgg16':
        import network.vgg16 as vgg16
        net = vgg16.Net()
    else:
        raise("wrong model settings")

    net.load_state_dict(torch.load('{}/weights/{}.pth'.format(
        args.root_dir, args.model)),
                        strict=False)

    return net


def spacial_norm_preds_only(mask, class_cur):
    temp = np.zeros(mask.shape)
    # spactial normalize
    num_class_cur = len(class_cur)
    temp_cur = mask[class_cur, :, :].reshape([num_class_cur, -1])
    # temp_min = np.min(temp_cur, axis=1, keepdims=True)
    # temp_cur = temp_cur - temp_min
    temp_cur[temp_cur < 0] = 0    # manual relu
    temp_max = np.max(temp_cur, axis=1, keepdims=True)
    temp_max[temp_max == 0] = 1
    temp_cur = temp_cur / temp_max

    if class_cur[0] == 0 and num_class_cur > 1:
        # temp_cur[0,:] = mask[0,:,:].reshape([1,-1]) - np.sum(temp_cur[1:,:], axis=0)
        temp_cur[0, np.sum(temp_cur[1:, :], axis=0) > 0.1] = 0

    temp[class_cur, :, :] = temp_cur.reshape([num_class_cur, mask.shape[1],
                                              mask.shape[2]])
    temp = temp * 0.9 + 0.05

    return temp


def net_outputs_to_list(layer4_feature_np, x_np, sm_mask_np, cur_class):
    # sm_mask_np need no process, already range in [0,1]
    # layer4_feature_np shape: (1,2048,41,41) range from [0, 17.xxxx], need normalization
    # x_np shape:(1,21,41,41) range from about [-56.38xxx, 130.51xxx], need normalization

    x_np_norm = np.expand_dims(spacial_norm_preds_only(x_np.squeeze(),
                                                       cur_class),
                               axis=0)
    x = layer4_feature_np.max(axis=1, keepdims=True)
    ly4 = layer4_feature_np/x

    # return [ly4, sm_mask_np[:,1:,:,:]] # [ly4, sm_mask_np[:,1:,:,:]]
    return [ly4, x_np_norm[:, 1:, :, :]]


def log_config(args):
    log_dir = os.path.join(args.root_dir, "/logs")
    logging.basicConfig(
         filename=os.path.join(
             log_dir, "{}_{}.log".format(args.id, args.mode)),
         level=logging.ERROR)
    logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # logger.addHandler(handler)
    return logger


def output_log(output_str, logger=None):
    """
    standard output and logging
    """
    print("[{}]: {}".format(datetime.datetime.now(), output_str))
    if logger is not None:
        logger.critical("[{}]: {}".format(datetime.datetime.now(), output_str))


def log_args(args, logger):
    '''
    log args
    '''
    attrs = [(p, getattr(args, p)) for p in dir(args) if not p.startswith('_')]
    for key, value in attrs:
        output_log("{}: {}".format(key, value), logger=logger)
