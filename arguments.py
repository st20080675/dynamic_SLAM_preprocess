import argparse


def get_args():
    parser = argparse.ArgumentParser(description='process_to_mask_moving_objects')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--CRF_model', type=str, default="my_CRF",  # resnet SEC
                        help='CRF model my_CRF|SEC_CRF|adaptive_CRF')
    parser.add_argument('--batch_size', type=int, default=15,
                        help='training batch size')
    parser.add_argument('--origin_size', action='store_true', default=False,
                        help='when it is training')
    parser.add_argument('--relu_mask', action='store_true', default=False,
                        help='whether apply relu for not')
    parser.add_argument('--preds_only', action='store_true', default=True,
                        help='whether only use')
    parser.add_argument('--input_size', nargs='+', type=int, default=[321, 321],
                        help='size of training images [224,224]|[321,321]')
    # 32 for 256 input; 28 for 224 input
    parser.add_argument('--output_size', nargs='+', type=int, default=[41, 41],
                        help='size of output mask')
    parser.add_argument('--num_classes', type=int, default=21,
                        help='classificiation nums')
    parser.add_argument('--no_bg', action='store_true', default=False,
                        help='no background')
    parser.add_argument('--color_vote', action='store_true', default=True,
                        help='no background')
    parser.add_argument('--fix_CRF_itr', action='store_true', default=False,
                        help='fix CRF iteration')
    parser.add_argument('--test_flag', action='store_true', default=False,
                        help='when it is training')
    parser.add_argument('--need_mask_flag', action='store_true', default=False,
                        help='need mask even training')

    parser.add_argument('--data_dir', type=str,
                        default="/home/sunting/Documents/semantic_SLAM/dataset/tum/dynamic_objects/rgbd_dataset_freiburg3_sitting_rpy/",
                        help='data loading directory')
    parser.add_argument('--data_file', type=str, default="rgb_name_only.txt",
                        help='.txt file contain images from subdir from data_dir')
    parser.add_argument('--kitti_image_folder', type=str, default="image_3",
                        help='the image folder')

    parser.add_argument('--use_depth', default=False,
                        help='whether use depth for CRF and dataloader')
    parser.add_argument('--depth_data_file', type=str, default="depth_name_only.txt",
                        help='.txt file contain depth images from subdir from data_dir')
    parser.add_argument('--data_set', type=str, default="TUM",
                        help='dataset')
    parser.add_argument('--associate_data_file', type=str, default="associate_list.txt",
                        help='.txt file contain associated images and depth from subdir from data_dir')


    parser.add_argument('--root_dir', type=str, default="/home/sunting/Documents/program/dynamic_SLAM_preprocess",
                        help='root dir')
    parser.add_argument('--model', type=str, default="resnet38",  # resnet SEC
                        help='model type resnet|SEC|vgg16|resnet38|vgg16_cuesec')
    parser.add_argument('--colorgray', type=str, default="color",
                        help='color|gray for cls')
    parser.add_argument('--save_path', type=str,
                        default="blurred",
                        help='save path')


    args = parser.parse_args()
    if args.no_bg is True:
        args.num_classes = 20


    return args
