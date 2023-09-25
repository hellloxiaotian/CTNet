import sys

import torch

sys.path.append('../')
import os
import numpy as np
from utils import utils_image
import torchvision.transforms as transforms
from itertools import chain
from torch.utils.data import Dataset
from tool.common_tools import add_noise
import random
import cv2
from tqdm import tqdm


def gen_patches(img, patch_size=48, n=128, aug=True, aug_plus=False):
    '''
    :param img: input_img
    :param patch_size:
    :param n: a img generate n patches
    :param aug: if need data augmentation or not
    :return: a list of patches
    '''

    patches = list()

    ih, iw, _ = img.shape

    ip = patch_size

    for _ in range(0, n):  # 一张图片产生n个patches
        iy = random.randrange(0, ih - ip + 1)
        ix = random.randrange(0, iw - ip + 1)

        # --------------------------------
        # get patch
        # --------------------------------
        patch = img[iy:iy + ip, ix:ix + ip, :]

        # --------------------------------
        # augmentation - flip, rotate
        # --------------------------------
        if aug:  # need augmentation
            if aug_plus:
                mode = random.randint(0, 6)
                f_aug = utils_image.augment_img_plus
            else:
                mode = random.randint(0, 7)
                f_aug = utils_image.augment_img
        else:  # don't need augmentation
            mode = 0
            f_aug = utils_image.augment_img

        patch = f_aug(patch, mode=mode)

        patches.append(patch)

    return patches


def train_data_generator(data_dir, save_dir, n_channels, patch_size, n_patches, aug_plus=False):

    # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
    filelist = os.listdir(data_dir)

    i = 0
    for img_name in tqdm(filelist):
        path_img = os.path.join(data_dir, img_name)

        # 打开一张图片
        img = utils_image.imread_uint(path_img, n_channels=n_channels)  # RGB

        # 1张图片产生Patches
        patches = gen_patches(img, patch_size=patch_size, n=n_patches, aug_plus=aug_plus)

        for patch in patches:
            img = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)  # RGB->BGR
            cv2.imwrite(save_dir + '_' + str(i) + img_name[-4:], img)
            i = i + 1

    return 0


if __name__ == '__main__':

    data_dir = '/data/zmh/dataset/data/images/train/CBSD432+pristine+DIV2k+Flick2k'
    # save_dir = '/data/zmh/dataset/data/images/train/CBSD432+pristine+DIV2K+Flick2k_p48_n1080/'
    save_dir = '/home/zhengmenghua/Project/DataSet/CBSD432+pristine+DIV2K+Flick2k_p48_n1080/'
    if not os.path.exists(save_dir):  # 没有该文件夹，则创建该文件夹
        os.makedirs(save_dir)
        print("Make the dir:{}".format(save_dir))

    train_data_generator(data_dir, save_dir, n_channels=3, patch_size=48, n_patches=1)
