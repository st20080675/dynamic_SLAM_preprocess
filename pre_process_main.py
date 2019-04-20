import torch
import torch.nn as nn
import numpy as np
from skimage.transform import resize
from arguments import get_args
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt
import ST_adaptive_CRF
import ST_CRF



if __name__=="__main__":
    args = get_args()
    sys.path.append(args.root_dir)

    flag_visual = True
    if args.origin_size or flag_visual:
        args.batch_size = 1

    from utils import log_args, output_log, log_config
    # logger = log_config(args)

    from utils import model_loader
    net_seg = model_loader(args)

    from data_loader import VOCData
    dataloader = VOCData(args)

    args.need_mask_flag = True
    args.origin_size = False

    flag_use_cuda = torch.cuda.is_available()

    if args.CRF_model == 'adaptive_CRF':
        st_crf_layer = ST_adaptive_CRF.STCRFLayer(True)
    else:
        st_crf_layer = ST_CRF.STCRFLayer(False)

    min_prob = 0.0001
    sm_layer = nn.Softmax2d()

    if flag_use_cuda:
        net_seg.cuda()
        sm_layer.cuda()

    net_seg.train(False)

    save_dir = os.path.join(args.data_dir, args.save_path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # log_args(args, logger)
    with torch.no_grad():
        for data in dataloader.dataloaders['data']:
            inputs = data['input']
            img = data['img']
            raw = data['raw']

            if flag_use_cuda:
                inputs = inputs.cuda()

            _, seg_x = net_seg.forward_cue(inputs)
            sm_mask = sm_layer(seg_x) + min_prob
            sm_mask = sm_mask / sm_mask.sum(dim=1, keepdim=True)

            mask_mended = sm_mask.detach().cpu().numpy()

            # result big shape
            result_big, result_small = st_crf_layer.run(mask_mended, img.numpy())

            for i in range(result_big.shape[0]):

                raw_image_big = resize(
                    np.transpose(result_big[i], (1, 2, 0)),
                    (raw[i].size(0), raw[i].size(1)),
                    order=0)
                mask_pre = np.argmax(raw_image_big, axis=2)


            if flag_visual:
                plt.subplot(1, 3, 1)
                plt.imshow(raw[0])
                plt.subplot(1, 3, 2)
                raw_mask_pred = np.argmax(mask_mended[0], axis=0)
                plt.imshow(raw_mask_pred)
                plt.subplot(1, 3, 3)
                plt.imshow(mask_pre)
                # plt.close('all')


                # Image.fromarray(mask_pre.astype(np.uint8)).save('{}/{}/{}.png'.format(args.data_dir, args.save_path, name))


