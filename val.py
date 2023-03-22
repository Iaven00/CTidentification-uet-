import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from PIL import Image
import archs
from dataset import Dataset, Testset
from metrics import iou_score
from utils import AverageMeter
"""
需要指定参数：--name dsb2018_96_NestedUNet_woDS
"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="data_NestedUNet_woDS",
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))  # 读取所有图片
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids] # 分离文件名和扩展名

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        A.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    for c in range(config['num_classes']):  # 创建输出文件夹
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meter.avg)
    
    plot_examples(input, target, model,num_examples=3)
    
    torch.cuda.empty_cache()

def predict(outpath , inpath):
    # inpath为需要预测结果的数据集，outpath为指定输出结果的路径
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cudnn.benchmark = True
    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)
    # create model
    # print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join(inpath, '*' + config['img_ext']))  # 读取所有图片
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]  # 分离文件名和扩展名

    val_img_ids = img_ids

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        A.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])


    val_dataset = Testset(
        img_ids=val_img_ids,
        img_dir=inpath,
        img_ext=config['img_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # avg_meter = AverageMeter()

    # 创建结果文件夹
    os.makedirs(outpath, exist_ok=True)

    with torch.no_grad():
        for input, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()

            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)


            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join(outpath, meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))


    torch.cuda.empty_cache()



def plot_examples(datax, datay, model ,num_examples=6):
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18, 4*num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx+1]).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0,:,:].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][2].set_title("Target image")
    plt.show()


if __name__ == '__main__':
    # main()
    predict('outputs/new1','inputs/data_test')