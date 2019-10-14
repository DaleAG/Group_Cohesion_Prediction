import os, sys, shutil
import random as rd
from os import listdir
from PIL import Image
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pdb
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import cv2
def load_imgs(image_list_file):
    imgs_first = list()
    imgs_second = list()
    imgs_third = list()

    with open(image_list_file, 'r') as imf:
        for line in imf:
            line = line.strip()
            img_path, img_label = line.split(' ',1)
            gr_em_label = img_path.split('/')[-1].split('_')[0]
            if gr_em_label =='neg':
                gr_em_label = 0
            if gr_em_label =='neu':
                gr_em_label = 1
            if gr_em_label =='pos':
                gr_em_label =2
            gr_em_label = int(gr_em_label)
            img_lists = os.listdir(img_path)
            img_lists.sort()
            random_num = np.zeros(3)
            random_num[0] = random.randint(0,int(len(img_lists)/3))
            random_num[1] = random.randint(int(len(img_lists)/3),int(2*len(img_lists)/3))
            try:
                random_num[2] = random.randint(int(2*len(img_lists)/3),len(img_lists)-1)
            except:
                pdb.set_trace()
            # pdb.set_trace()
            face_path_first = img_path + '/' + img_lists[int(random_num[0])]
            face_path_second = img_path + '/' + img_lists[int(random_num[1])]
            face_path_third = img_path + '/' + img_lists[int(random_num[2])]
            label = float(int(img_label)/3)
            #pdb.set_trace()            
            imgs_first.append((face_path_first,label,gr_em_label))
            imgs_second.append((face_path_second,label,gr_em_label))
            imgs_third.append((face_path_third,label,gr_em_label))

                
    return imgs_first,imgs_second,imgs_third

class MsCelebDataset(data.Dataset):
    def __init__(self, image_list_file, transform=None):
        self.imgs_first, self.imgs_second, self.imgs_third = load_imgs(image_list_file)
        self.transform = transform

    def __getitem__(self, index):
        #pdb.set_trace()
        path_first, target_first, gr_em_label = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)


        path_second, target_second, gr_em_label = self.imgs_second[index]
        img_second = Image.open(path_second).convert("RGB")
        if self.transform is not None:
            img_second = self.transform(img_second)


        path_third, target_third, gr_em_label = self.imgs_third[index]
        img_third = Image.open(path_third).convert("RGB")
        if self.transform is not None:
            img_third = self.transform(img_third)



        return img_first, target_first ,gr_em_label, img_second,target_second,gr_em_label, img_third,target_third, gr_em_label
    def __len__(self):
        return len(self.imgs_first)




class CaffeCrop(object):
    """
    This class take the same behavior as sensenet
    """
    def __init__(self, phase):
        assert(phase=='train' or phase=='test')
        self.phase = phase

    def __call__(self, img):
        # pre determined parameters
        final_size = 224
        final_width = final_height = final_size
        crop_size = 110
        crop_height = crop_width = crop_size
        crop_center_y_offset = 15
        crop_center_x_offset = 0
        if self.phase == 'train':
            scale_aug = 0.02
            trans_aug = 0.01
        else:
            scale_aug = 0.0
            trans_aug = 0.0
        
        # computed parameters
        randint = rd.randint
        scale_height_diff = (randint(0,1000)/500-1)*scale_aug
        crop_height_aug = crop_height*(1+scale_height_diff)
        scale_width_diff = (randint(0,1000)/500-1)*scale_aug
        crop_width_aug = crop_width*(1+scale_width_diff)


        trans_diff_x = (randint(0,1000)/500-1)*trans_aug
        trans_diff_y = (randint(0,1000)/500-1)*trans_aug


        center = ((img.width/2 + crop_center_x_offset)*(1+trans_diff_x),
                 (img.height/2 + crop_center_y_offset)*(1+trans_diff_y))

        
        if center[0] < crop_width_aug/2:
            crop_width_aug = center[0]*2-0.5
        if center[1] < crop_height_aug/2:
            crop_height_aug = center[1]*2-0.5
        if (center[0]+crop_width_aug/2) >= img.width:
            crop_width_aug = (img.width-center[0])*2-0.5
        if (center[1]+crop_height_aug/2) >= img.height:
            crop_height_aug = (img.height-center[1])*2-0.5

        crop_box = (center[0]-crop_width_aug/2, center[1]-crop_height_aug/2,
                    center[0]+crop_width_aug/2, center[1]+crop_width_aug/2)

        mid_img = img.crop(crop_box)
        res_img = img.resize( (final_width, final_height) )
        return res_img
