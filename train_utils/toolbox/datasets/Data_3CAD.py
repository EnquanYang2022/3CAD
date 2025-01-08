import glob

import torch
import os
import math
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .data_utils import perlin_noise
from torchvision.transforms import InterpolationMode
class Data_3CAD(Dataset):
    def __init__(self, data_path, dtd_path,class_name=None,phase='train',resize=256, cropsize=256):

        self.data_path = data_path
        self.dtd_path = dtd_path
        self.class_name = class_name
        self.phase = phase
        self.resize = resize
        self.cropsize = cropsize
        self.transform_size = transforms.Compose([
            transforms.Resize([resize,resize]),
            ])
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.transform_mask = transforms.Compose([
            transforms.Resize([resize,resize], interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()])

        self.class_in_task = []
        #my_data
        self.imgs_list = []
        self.labels_list = []
        self.masks_list = []
        #dtd
        self.dtd_imgs_list = []


        # mark each sample task id
        self.sample_num_in_task = []
        self.sample_indices_in_task = []

        # load dataset
        self.load_dataset()
        self.allocate_task_data()

    def __getitem__(self, idx):
        img_src, label, mask = self.imgs_list[idx], self.labels_list[idx], self.masks_list[idx]
        dtd_img_src = self.dtd_imgs_list[idx%len(self.dtd_imgs_list)]
        dtd_img = Image.open(dtd_img_src).convert('RGB')
        dtd_img = dtd_img.resize([self.resize,self.resize],Image.BILINEAR)
        img = Image.open(img_src).convert('RGB')
        img = self.transform_size(img)

        if self.phase=='train':
            aug_image, aug_mask = perlin_noise(img, dtd_img, aug_prob=1.0)
            aug_image = self.transform_img(aug_image)
            img_origin = self.transform_img(img)
            return {
                'img_origin':img_origin,'img_aug': aug_image, 'mask': aug_mask,
            }
        else:
            img = self.transform_img(img)
            if label == 0:
                if isinstance(img, tuple):
                    mask = torch.zeros([1, img[0].shape[1], img[0].shape[2]])
                else:
                    mask = torch.zeros([1, img.shape[1], img.shape[2]])
            else:
                mask = Image.open(mask)
                mask = self.transform_mask(mask)
            return {
                'img': img, 'label': label, 'mask': mask, 'img_src': img_src,
            }



    def __len__(self):
        return len(self.imgs_list)

    def load_dataset(self):
        # get data
        x, y, mask = [], [], []

        img_dir = os.path.join(self.data_path, self.class_name, self.phase)
        gt_dir = os.path.join(self.data_path, self.class_name, 'ground_truth')

        dtd_img_list = sorted(glob.glob(self.dtd_path+'/*/*.jpg'))

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_path_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)
                                    if f.endswith('.png')])
            x.extend(img_path_list)

            if img_type == 'good':
                y.extend([0] * len(img_path_list))
                mask.extend([None] * len(img_path_list))
            else:
                y.extend([1] * len(img_path_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_name_list = [os.path.splitext(os.path.basename(f))[0] for f in img_path_list]
                gt_path_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                for img_fname in img_name_list]
                mask.extend(gt_path_list)


        self.sample_num_in_task.append(len(x))
        self.dtd_imgs_list.extend(dtd_img_list)
        self.imgs_list.extend(x)
        self.labels_list.extend(y)
        self.masks_list.extend(mask)


    def allocate_task_data(self):
        start = 0
        for num in self.sample_num_in_task:
            end = start + num
            indice = [i for i in range(start, end)]
            random.shuffle(indice)
            self.sample_indices_in_task.append(indice)
            start = end


if __name__ == '__main__':
    train_dataset = Data_3CAD(data_path='./my_data/',dtd_path='./dtd/images/',class_name='aluminum_Camera_Cover',phase='test')
    kwargs = ({"num_workers": 8, "pin_memory": True} if torch.cuda.is_available() else {})
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True,**kwargs)
    for _,d in enumerate(train_loader):

        print(d['img'].shape) #16,3,256,256