import copy
import os, time

import imageio
import numpy as np
import torch
import torchvision
from PIL import Image
from cv2 import resize
from matplotlib import pyplot, image
import cv2
from scipy import misc

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torchvision import transforms

from buster_train import ModelWithLoss

from net_cmsd import SimilarityNet
# from small_Unet import U2NET

from dct_bd_net import Dct_bd_net

from Unet import UNet
# from U2net import U2NET
from vu2net_doa_2_2 import VGG19



input_size = 256





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = VGG19()

model = model.eval()




def fs_(pred, gt):
    ref = gt.flatten() == 255.
    hyf = pred.flatten() >= 0.5
    # hyf = pred.flatten() == 1
    ac, re, fscore, _ = precision_recall_fscore_support(
    ref, hyf, pos_label=1, average='binary')
    auc = roc_auc_score(ref, hyf)
    return ac, re, fscore, auc

imgslist = os.listdir('D:/code/CSFD/oringe')
acc = []
ree = []
fss = []
auc = []
transform = torchvision.transforms.Compose([transforms.ToPILImage(),transforms.Resize((320, 320),interpolation=2),
                                                 torchvision.transforms.ToTensor(),
                                            # transforms.ColorJitter(brightness=0.3, contrast=0.3),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225]),
                                            ])



def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            imagelist.append(os.path.join(parent, filename))
        return imagelist



path1 = './log_corr_dense/'

quanzhong_list = get_img_file(path1)
# print(quanzhong_list)

result_list = []
result_list_au = []
result_list_re = []



for index ,path in enumerate(quanzhong_list):
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.eval()

    acc = []
    ree = []
    fss = []
    auc = []


    for index in range(1, 201):

        # imag = Image.open('D:/代码/CSFD/oringe/' +  str(index+1) + ".tif")

        if index < 10:
            start = str(0) + str(0)
        elif index < 100:
            start = str(0)
        else:
            start = str("")
        image = cv2.imread('D:/code/CSFD/CoMoFoD_small_v2/' + start + str(index) + "_F" + ".png")

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # pyplot.imshow(img)
        # pyplot.show()

        dct_image = cv2.imread('D:/dataset/dct_comofod/' + "dct_image_" + str(index) + ".png")




        img = transform(img)
        img = np.expand_dims(img, axis=0)

        img = torch.from_numpy(img).cuda()

        pred, _, _, _, _ = model(img)
        # print(pred.shape)



        pred = pred.squeeze(0)
        # print(pred[1])



        pred = pred.squeeze(0)


        pred = pred.cpu().detach().numpy()
        # pyplot.imshow(pred)
        # pyplot.show()

        gt = cv2.imread('D:/code/CSFD/CoMoFoD_small_v2/' + start + str(index) + "_B" + ".png", flags=0)
        # gt = gt.convert("L")
        gt = resize(gt,(320, 320),interpolation=1)

        # pyplot.imshow(gt)
        # pyplot.show()


        ac, re, fs, au = fs_(pred, gt)
        acc.append(ac)
        ree.append(re)
        fss.append(fs)
        auc.append(au)
        del img, pred, gt

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