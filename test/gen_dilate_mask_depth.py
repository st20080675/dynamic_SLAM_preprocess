import numpy as np
from skimage.transform import resize
from arguments import get_args
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

def get_image_list(data_dir, list_file):
     with open(os.path.join(data_dir, list_file), "r") as f:
        all_files = f.readlines()
        return [item.split() for item in all_files]


if __name__=="__main__":
    args = get_args()
    sys.path.append('./..')
    flag_visualization = True
    args.data_file = "associate_list.txt"
    args.data_dir = "/home/sunting/Documents/semantic_SLAM/dataset/tum/dynamic_objects/rgbd_dataset_freiburg3_walking_xyz/"

    seg_mask_path = 'mask_w_color_depth'
    save_masked_depth_path = 'masked_dilated_w_color_depth'
    move_class = [15] # [5, 9, 15]


    file_list = get_image_list(args.data_dir, args.data_file)

    for item in file_list:
        depth_name = os.path.join(args.data_dir, item[3])

        mask_name = os.path.join(args.data_dir, seg_mask_path, item[1])
        # img = Image.open(img_name)
        # img_blur = img.filter(ImageFilter.BLUR)
        depth = ndimage.imread(depth_name)
        mask = ndimage.imread(mask_name)

        mask_combine = np.zeros(mask.shape, dtype='uint8')

        for i_class in move_class:
            mask_combine[mask==i_class] = 1

        struct = ndimage.generate_binary_structure(2, 2)
        mask_combine = ndimage.binary_dilation(mask_combine, structure = struct ,iterations=2).astype(mask_combine.dtype)
        depth[mask_combine>0] = 0

        if flag_visualization:
            plt.subplot(1,3,1)
            plt.imshow(mask)
            plt.subplot(1,3,2)
            plt.imshow(mask_combine)
            plt.subplot(1,3,3)
            plt.imshow(depth)

        depth_masked = Image.fromarray(depth)
        depth_masked.save('{}{}/{}'.format(args.data_dir, save_masked_depth_path, i_img))



