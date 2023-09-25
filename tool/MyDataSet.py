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
import pdb
from sys import getrefcount


class Art_nosie_Dataset(Dataset):   # 添加人工噪声

    def __init__(self, args, data_dir, mode='train', ori_image_size=True):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        """
        self.args = args
        self.mode = mode
        self.patch_size = args.patch_size
        self.nose_level = args.sigma
        self.n_patches = args.n_pat_per_image
        self.n_channels = args.n_colors
        self.ori_image_size = ori_image_size
        self.name_list = None
        self.mode = mode
        if mode == 'train':
            self.data_lsit = self.train_data_generator(data_dir)  # patches 的集合
        else: # mode == test
            self.data_lsit, self.name_list = self.test_data_generator(data_dir)

    def __getitem__(self, index):
        clean_img = self.data_lsit[index]

        # ----------------------------------------------------------------------
        # HWC to CHW, numpy(uint) to tensor
        # ----------------------------------------------------------------------
        clean_img = utils_image.uint2tensor3(clean_img).mul(self.args.rgb_range/255.)

        # add noise
        if self.nose_level != 100:  # 噪声水平确定
            nos_img = add_noise(clean_img, noise_leve=self.nose_level, rgb_range=self.args.rgb_range)
        else:  # nose_level == 100 表示 盲降噪
            nos_img = add_noise(clean_img, noise_leve=np.random.randint(0, 55, 1)[0], rgb_range=self.args.rgb_range)

        if self.name_list is None:
            return clean_img, nos_img
        else:
            return clean_img, nos_img, self.name_list[index]

    def __len__(self):
        return len(self.data_lsit)

    def gen_patches(self, img, patch_size=48, n=128, aug=True, aug_plus=False):
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

        for _ in range(0, n):   # 一张图片产生n个patches
            iy = random.randrange(0, ih - ip + 1)
            ix = random.randrange(0, iw - ip + 1)

            # --------------------------------
            # get patch
            # --------------------------------
            patch = img[iy:iy+ip, ix:ix+ip, :]

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

    def train_data_generator(self, data_dir):
        # 用于存储所有图片
        img_list = list()

        # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
        filelist = os.listdir(data_dir)

        for img_name in filelist:
            path_img = os.path.join(data_dir, img_name)

            # 打开一张图片
            img = utils_image.imread_uint(path_img, n_channels=self.n_channels)  # RGB

            # 1张图片产生Patches
            patches = self.gen_patches(img,
                                       patch_size=self.patch_size,
                                       n=self.n_patches,
                                       aug_plus=self.args.aug_plus)

            # 将patches加入到img_lsit中
            img_list.append(patches)

        # img_lsit 为所有patches的集合
        img_list = list(chain(*img_list))

        return img_list

    def test_data_generator(self, data_dir):
        # 用于存储所有图片
        img_list = list()
        name_list = list()

        # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
        filelist = os.listdir(data_dir)

        filelist.sort()

        for img_name in filelist:
            path_img = os.path.join(data_dir, img_name)

            # 打开一张图片
            img = utils_image.imread_uint(path_img, n_channels=self.n_channels)  # RGB
            # print('=====================================')

            # 测试图片返回 完整图片
            img_list.append(img)
            name_list.append(img_name)

        return img_list, name_list


class Real_Dataset(Dataset):    # 真实噪声

    def __init__(self, args, data_dir, mode='train'):
        """
        :param data_dir: str, 数据集所在路径
        """
        self.args = args
        self.mode = mode
        self.patch_size = args.patch_size
        self.n_patches = args.n_pat_per_image
        self.n_channels = args.n_colors
        self.name_list = None
        if self.mode == 'train':
            self.gt_lsit, self.real_list = self.train_data_generator(data_dir)  # patches 的集合
        elif self.mode == 'test':
            self.gt_lsit, self.real_list, self.name_list = self.test_data_generator(data_dir)  # patches 的集合

    def __getitem__(self, index):
        clean_img = self.gt_lsit[index]
        real_img = self.real_list[index]

        # ---------------------------------------
        # HWC to CHW, numpy(uint) to tensor
        # ---------------------------------------

        clean_img = utils_image.uint2tensor3(clean_img).mul(self.args.rgb_range/255.)
        real_img = utils_image.uint2tensor3(real_img).mul(self.args.rgb_range / 255.)

        if self.name_list is None:
            return clean_img, real_img
        else:
            return clean_img, real_img, self.name_list[index]

    def __len__(self):
        return len(self.gt_lsit)

    def gen_patches(self, img1, img2, patch_size=48, n=128, aug=True, aug_plus=False):

        '''
        :param img: input_img
        :param patch_size:
        :param n: a img generate n patches
        :param aug: if need data augmentation or not
        :return: a list of patches
        '''

        patches1 = list()
        patches2 = list()

        ih, iw, _ = img1.shape

        ip = patch_size

        for _ in range(0, n):   # 一张图片产生n个patches
            iy = random.randrange(0, ih - ip + 1)
            ix = random.randrange(0, iw - ip + 1)

            # --------------------------------
            # get patch
            # --------------------------------
            patch1 = img1[iy : iy+ip, ix : ix+ip, :]
            patch2 = img2[iy : iy+ip, ix : ix+ip, :]

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

            patch1 = f_aug(patch1, mode=mode)
            patch2 = f_aug(patch2, mode=mode)

            patches1.append(patch1)
            patches2.append(patch2)

        return patches1, patches2

    def train_data_generator(self, data_dir):
        # 用于存储所有图片
        real_img_list = list()
        gt_img_list = list()

        # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
        reallist = os.listdir(os.path.join(data_dir, "noise"))
        gtlist = os.listdir(os.path.join(data_dir,  "clean"))

        # 对文件名进行排序，listdir()获得的list是无序的
        reallist.sort()
        gtlist.sort()

        for realimage, gtimage in zip(reallist, gtlist):

            real_path = os.path.join(data_dir, "noise", realimage)
            gt_path = os.path.join(data_dir, "clean", gtimage)

            # 打开一张图片
            real_img = utils_image.imread_uint(real_path, n_channels=self.n_channels)  # RGB
            gt_img = utils_image.imread_uint(gt_path, n_channels=self.n_channels)  # RGB

            # 1张图片产生Patches

            real_patches, gt_patches = self.gen_patches(real_img, gt_img, patch_size=self.patch_size, n=self.n_patches,
                                            aug_plus=self.args.aug_plus)
            # gt_patches = self.gen_patches(gt_img, patch_size=self.patch_size, n=self.n_patches,
            #                                aug_plus=self.args.aug_plus)

            # 将patches加入到img_lsit中
            real_img_list.append(real_patches)
            gt_img_list.append(gt_patches)

        # img_lsit 为所有patches的集合
        real_img_list = list(chain(*real_img_list))
        gt_img_list = list(chain(*gt_img_list))

        return gt_img_list, real_img_list

    def test_data_generator(self, data_dir): # 和train_data_generator的区别主要在于不用分割patch
        # 用于存储所有图片
        real_img_list = list()
        gt_img_list = list()
        name_list = list()

        # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
        reallist = os.listdir(os.path.join(data_dir, "real"))
        gtlist = os.listdir(os.path.join(data_dir, "mean"))

        # 对文件名进行排序，listdir()获得的list是无序的
        reallist.sort()
        gtlist.sort()

        for realimage, gtimage in zip(reallist, gtlist):
            real_path = os.path.join(data_dir, "real", realimage)
            gt_path = os.path.join(data_dir, "mean", gtimage)

            # 打开一张图片
            real_img = utils_image.imread_uint(real_path, n_channels=self.n_channels)  # RGB
            gt_img = utils_image.imread_uint(gt_path, n_channels=self.n_channels)  # RGB

            '''
            # 1张图片产生Patches
            real_patches = self.gen_patches(real_img, patch_size=self.patch_size, n=self.n_patches,
                                            aug_plus=self.args.aug_plus)
            gt_patches = self.gen_patches(gt_img, patch_size=self.patch_size, n=self.n_patches,
                                          aug_plus=self.args.aug_plus)
            '''

            # 将patches加入到img_lsit中
            real_img_list.append(real_img)
            gt_img_list.append(gt_img)
            name_list.append(realimage)

        return gt_img_list, real_img_list, name_list


class GenDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_dir, mode, ori_image_size=True):

        '''
        with open(data_path, "r") as f:
            self.data = f.readlines()
            # 如果这里都爆内存的话，
            # 看起来只能使用文件指针，在getitem里边逐行读取了
            # 得到的data是 list[str]
        '''
        self.args = args
        self.mode = mode
        self.patch_size = args.patch_size
        self.nose_level = args.sigma
        self.n_patches = args.n_pat_per_image
        self.n_channels = args.n_colors
        self.ori_image_size = ori_image_size
        self.name_list = None
        # 用于存储所有图片
        self.img_path_list = list()

        # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
        filelist = os.listdir(data_dir)

        for img_name in filelist:
            path_img = os.path.join(data_dir, img_name)
            self.img_path_list.append(path_img)

        # random.shuffle(self.data)
        self.data_gen = self.get_data()

    def get_data(self):

        img_list = list()

        for path_img in self.img_path_list:
            # 打开一张图片
            img = utils_image.imread_uint(path_img, n_channels=self.n_channels)  # RGB
            # 1张图片产生Patches

            patches = self.gen_patches(img, patch_size=self.patch_size, n=self.n_patches, aug_plus=self.args.aug_plus)
            img_list.append(patches)

        img_list = list(chain(*img_list))
        random.shuffle(img_list)
        
        while len(img_list) > 0:
            # 逐个把数据返回,每次只返回一条
            clean_img = img_list.pop()
            clean_img = utils_image.uint2tensor3(clean_img).mul(self.args.rgb_range / 255.)

            # noise_img = add_noise(clean_img, noise_leve=self.nose_level, rgb_range=self.args.rgb_range)
            # add noise
            if self.nose_level != 100:  # 噪声水平确定
                noise_img = add_noise(clean_img, noise_leve=self.nose_level, rgb_range=self.args.rgb_range)
            else:  # nose_level == 100 表示 盲降噪
                noise_img = add_noise(clean_img, noise_leve=np.random.randint(0, 55, 1)[0], rgb_range=self.args.rgb_range)

            yield clean_img, noise_img

    def __len__(self):
        # 这里返回长度是用于tqdm进度条显示用的
        # 我这里乘以4是我之前预处理的时候看得到总量大概是文档数目的4倍
        # 你也可以设定一个很大的数字，当dataloader提取不到数据的时候就会停止
        return len(self.img_path_list) * self.n_patches

    def __getitem__(self, idx):
        # 每次使用next函数返回生成器生成的一条数据，此处的idx用不到了
        return next(self.data_gen)

    def gen_patches(self, img, patch_size=48, n=128, aug=True, aug_plus=False):
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

        for _ in range(0, n):   # 一张图片产生n个patches
            iy = random.randrange(0, ih - ip + 1)
            ix = random.randrange(0, iw - ip + 1)

            # --------------------------------
            # get patch
            # --------------------------------
            patch = img[iy:iy+ip, ix:ix+ip, :]

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