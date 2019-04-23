from skimage.transform import resize
import matplotlib.pyplot as plt
import sys
import os
from arguments import get_args
from PIL import Image
from PIL import ImageFilter
import scipy.ndimage as ndimage
import time


def get_image_list(data_dir, list_file):
     with open(os.path.join(data_dir, list_file), "r") as f:
        all_files = f.readlines()
        return [item.split()[0] for item in all_files]


if __name__=="__main__":
    args = get_args()
    sys.path.append('./..')
    flag_visual = False

    file_list = get_image_list(args.data_dir, args.data_file)
    args.save_path = "blurred"

    start = time.time()
    for i in file_list:
        img_name = os.path.join(args.data_dir, i)
        # img = Image.open(img_name)
        # img_blur = img.filter(ImageFilter.BLUR)

        img = ndimage.imread(img_name)
        img_blur = ndimage.gaussian_filter(img, sigma=(7, 7, 0), order=0)

        if flag_visual:
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(img_blur)
            # plt.close('all')

        img_blur = Image.fromarray(img_blur)
        # img_blur.save('{}{}/{}'.format(args.data_dir, args.save_path, i))

    end = time.time()
    print(end - start)
    print("total number of images: {}".format(len(file_list)))
