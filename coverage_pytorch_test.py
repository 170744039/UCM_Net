
import os, time

import imageio
import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot, image
from cv2 import resize
import cv2
from numpy import split

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torchvision import transforms

from vu2net_2_dense import VGG19


input_size = 320



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model = VGG19()

model.eval()


def fs_(pred, gt):
    ref = gt.flatten() == 255
    hyf = pred.flatten() >= 0.5
    ac, re, fscore, _ = precision_recall_fscore_support(
    ref, hyf, pos_label=1, average='binary')
    auc = roc_auc_score(ref, hyf)
    return ac, re, fscore, auc

imgslist = os.listdir('D:/code/CSFD/oringe')
acc = []
ree = []
fss = []
auc = []
transform = torchvision.transforms.Compose([transforms.ToPILImage(),transforms.Resize((320, 320), interpolation=2),
                                                 torchvision.transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225]),
                                            ])



def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            imagelist.append(os.path.join(parent, filename))
        return imagelist



path1 = './log_vunet_dense/'

quanzhong_list = get_img_file(path1)
# print(quanzhong_list)

result_list = []
result_list_au = []
result_list_re = []


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist



path1 = 'D:/code/CSFD/oringe/'
path2 = 'D:/code/CSFD/image_zhu_2mask/'
image_list = get_img_file(path1)
mask_list = get_img_file(path2)
i = 1



for index ,path in enumerate(quanzhong_list):
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.eval()

    acc = []
    ree = []
    fss = []
    auc = []





    for index in range(0, 100):

        # imag = Image.open('D:/代码/CSFD/oringe/' +  str(index+1) + ".tif")

            list = int(str.split(str.split(image_list[index], sep='.')[0], sep="/")[-1])
            img = cv2.imread('E:/BaiduNetdiskDownload/image/' + str(list) + "t" + ".tif")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # pyplot.imshow(img)
            # pyplot.show()







            img = transform(img)
            img = np.expand_dims(img, axis=0)


            # imag = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
            img = torch.from_numpy(img).cuda()


            # pred = model(img)
            # pred, _, _, _, _ = model(img)
            pred, _, _, _, _, _,_= model(img)

            pred = pred.squeeze(0)
            pred = pred.squeeze(0)
            pred = pred.cpu().detach().numpy()
            # pyplot.imshow(pred)
            # pyplot.show()


            gt = cv2.imread('D:/code/CSFD/img_mask_100/' + "img_" + str(list - 1) + ".png", flags=0)
            # gt = imageio.imread(mask_list[index])
            # gt = gt[..., 0]
            # gt = gt.convert("L")
            gt = resize(gt, (320, 320), interpolation=1)

            # pyplot.imshow(gt)
            # pyplot.show()


            ac, re, fs, au = fs_(pred, gt)
            acc.append(ac)
            ree.append(re)
            # if fs > 0.1:
            #     fss.append(fs)
            #     i = i+1
            #     print(fs)
            fss.append(fs)
            auc.append(au)


            # auc.append(au)
            del img, pred, gt

    # print(i)
    result = path + "-------"+ str(np.mean(fss))
    result_au = path + "++++++++++++++++++" + str(np.mean(acc))
    result_re = path + "++++++++++++++++++" + str(np.mean(ree))

    result_list.append(result)
    result_list_au.append(result_au)
    result_list_re.append(result_re)

for i in range(100):
    print(result_list[i])
    print(result_list_au[i])
    print(result_list_re[i])

