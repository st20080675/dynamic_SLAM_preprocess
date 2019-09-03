import numpy as np
from skimage.transform import resize
from arguments import get_args
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import glob
import time

def get_TUM_image_list(data_dir, list_file):
     with open(os.path.join(data_dir, list_file), "r") as f:
        all_files = f.readlines()
        return [item.split() for item in all_files]

def get_KITTI_image_list(data_dir, kitti_image_folder):
    data_path = os.path.join(data_dir, kitti_image_folder)
    file_list = glob.glob(data_path+'/*')
    file_list.sort()
    return file_list

if __name__=="__main__":
    args = get_args()
    sys.path.append('./..')
    flag_visualization = False
    flag_mask_method = 'gray' # blur (gaussain)

    args.data_set = 'KITTI'  #'TUM' # 'KITTI'
    # args.data_dir = "/home/sunting/Documents/semantic_SLAM/dataset/tum/dynamic_objects/rgbd_dataset_freiburg3_walking_static_validation/"
    # args.data_dir = "/data_shared/Docker/tsun/docker/program/dynamic_SLAM_preprocess/data/KITTI/01"
    args.data_dir = "/media/sunting/sun/kitti_sequence/04/"
    args.kitti_image_folder = "image_0"

    blur_img_path = 'blurred'
    seg_mask_path = 'mask_w_color'
    save_masked_path = 'masked_w_color'
    move_class = [15] # [5, 9, 15]
    start_time = time.time()
    if args.data_set == 'KITTI':
        file_list = get_KITTI_image_list(args.data_dir, args.kitti_image_folder)
        move_class = [1,2,6,7,14,15]
        seg_mask_path = 'mask_w_depth'
        save_masked_path = 'masked_w_depth'
        save_dir = os.path.join(args.data_dir, save_masked_path, args.kitti_image_folder)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        print(len(file_list))
        for i_img in file_list:
            img_name = os.path.join(args.data_dir, i_img)
            mask_name = os.path.join(args.data_dir, seg_mask_path, i_img[-18:])

            img = ndimage.imread(img_name)

            mask = ndimage.imread(mask_name)

            mask_combine = np.zeros(mask.shape, dtype='uint8')

            for i_class in move_class:
                mask_combine[mask==i_class] = 1

            rows, cols = mask.shape
            gray_pixel = np.array([128], dtype='uint8')
            img[mask_combine>0] = gray_pixel

            if flag_visualization:
                plt.subplot(1,3,1)
                plt.imshow(mask)
                plt.subplot(1,3,2)
                plt.imshow(mask_combine)
                plt.subplot(1,3,3)
                plt.imshow(img)

            img_masked = Image.fromarray(img)
            img_masked.save('{}/{}/{}'.format(args.data_dir, save_masked_path, i_img[-18:]))


    elif args.data_set == 'TUM':
        file_list = get_TUM_image_list(args.data_dir, args.associate_data_file)
        save_dir = os.path.join(args.data_dir, save_masked_path, 'rgb')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        for i_img in file_list:
            img_name = os.path.join(args.data_dir, i_img[1])
            mask_name = os.path.join(args.data_dir, seg_mask_path, i_img[1])

            if flag_mask_method == 'blur':  # not used anymore
                blurred_img_name = os.path.join(args.data_dir, blur_img_path, i_img)
                blur_img = ndimage.imread(blurred_img_name)

            # img = Image.open(img_name)
            # img_blur = img.filter(ImageFilter.BLUR)
            img = ndimage.imread(img_name)

            mask = ndimage.imread(mask_name)

            mask_combine = np.zeros(mask.shape, dtype='uint8')

            for i_class in move_class:
                mask_combine[mask==i_class] = 1

            rows, cols = mask.shape
            gray_pixel = np.array([128, 128, 128], dtype='uint8')
            for i in range(0, rows):
                for j in range(0, cols):
                    if mask_combine[i, j] > 0:
                        if flag_mask_method == 'blur':
                            img[i, j, :] = blur_img[i, j, :]
                        else:
                            img[i, j, :] = gray_pixel

            if flag_visualization:
                plt.subplot(1,3,1)
                plt.imshow(mask)
                plt.subplot(1,3,2)
                plt.imshow(mask_combine)
                plt.subplot(1,3,3)
                plt.imshow(img)

            img_masked = Image.fromarray(img)
            img_masked.save('{}{}/{}'.format(args.data_dir, save_masked_path, i_img[1]))

    print(time.time()-start_time)




