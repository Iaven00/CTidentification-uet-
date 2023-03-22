import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
# def main():
#
#     imagepaths = glob('inputs/dsb2018_96/images/*')
#     maskpaths = glob('inputs/dsb2018_96/masks/0/*')
#     os.makedirs('inputs/stage_train', exist_ok=True)
#
#     for i in tqdm(range(len(imagepaths))):
#         imagepath = imagepaths[i]
#         maskpath = maskpaths[i]
#         os.makedirs(os.path.join('inputs/stage_train/', str(i)), exist_ok=True)
#         os.makedirs(os.path.join('inputs/stage_train/', str(i)+'/images'), exist_ok=True)
#         os.makedirs(os.path.join('inputs/stage_train/', str(i)+'/masks'), exist_ok=True)
#         img = cv2.imread(imagepath)
#         mask = cv2.imread(maskpath)
#
#
#         cv2.imwrite(os.path.join('inputs/stage_train/' + str(i) + '/images' , str(i) + '.png'),img)
#         cv2.imwrite(os.path.join('inputs/stage_train/' + str(i)+ '/masks' , str(i) + '.png'),mask)
#

def main():

    source_imgs_dir = 'inputs/dsb2018_96/images/'
    source_masks_dir = 'inputs/dsb2018_96/masks/0/'
    target_imgs_dir = 'inputs/dsb2022_128/images/'
    target_masks_dir = 'inputs/dsb2022_128/masks/0/'
    target_imgs_testdir = 'inputs/dsb2022_128test/images/'
    target_masks_testdir = 'inputs/dsb2022_128test/masks/0/'

    sid = os.listdir(source_imgs_dir)
    sid1, sid2 = train_test_split(sid, test_size=0.2, random_state=41)  # 训练集和测试集
    smd = os.listdir(source_masks_dir)
    smd1, smd2 = train_test_split(smd, test_size=0.2, random_state=41)  # 训练集和测试集

    for file in sid1:
        im = Image.open(source_imgs_dir + file)
        out = im.resize((256, 256), Image.ANTIALIAS)
        out.save(target_imgs_dir + file)

    for file in smd1:
        im = Image.open(source_masks_dir + file)
        out = im.resize((256, 256), Image.ANTIALIAS)
        out.save(target_masks_dir + file)

    # for file in sid2:
    #     im = Image.open(source_imgs_dir + file)
    #     out = im.resize((256, 256), Image.ANTIALIAS)
    #     out.save(target_imgs_testdir + file)
    #
    # for file in smd2:
    #     im = Image.open(source_masks_dir + file)
    #     out = im.resize((256, 256), Image.ANTIALIAS)
    #     out.save(target_masks_testdir + file)

    for filename in sid2:
        img = cv2.imread(source_imgs_dir + filename)
        # 保存图片
        cv2.imwrite(target_imgs_testdir + "/" + filename, img)

    for filename in smd2:
        img = cv2.imread(source_masks_dir + filename)
        # 保存图片
        cv2.imwrite(target_masks_testdir + "/" + filename, img)

if __name__ == '__main__':
    main()