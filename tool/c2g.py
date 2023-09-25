import os
import cv2
import sys
import numpy as np

def read_path(file_pathname):
    for filename in os.listdir(file_pathname):                        # 返回这一路径下的所有文件名，得到的是一个列表
        # print(filename)                                               # 这里print一下是为了验证os.listdir返回的不是绝对路径
        img = cv2.imread(file_pathname+'/'+filename, 1)               # 在前面要加上文件夹的路径，1是将图片按照彩色图来读
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)              # 彩色图转灰度图
        cv2.imwrite(file_pathname+'_G/' + "G_"+filename, gray_img)     # 保存灰度图；此处如果不改文件名，会覆盖原先的文件


read_path(sys.argv[1])
