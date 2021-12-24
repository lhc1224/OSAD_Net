import os
from PIL import Image
import cv2
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import numpy as np
import random

from utils.transform import load_image,load_label,\
    comput_mask,cv_random_crop_flip_ref,cv_center_crop,\
    cv_random_crop_flip,load_image_test,randomGaussianBlur,Normalize,load_pose

class ImageDataTrain(data.Dataset):
    def __init__(self,image_size,img_root,mask_root,ref,lst_path,txt_path):

        self.img_root=img_root
        self.mask_root=mask_root

        self.ref = ref
        self.img_size=image_size
        self.txt_path=txt_path

        with open(lst_path, 'r') as f:
            self.files = [x.strip() for x in f.readlines()]
        random.shuffle(self.files)

        self.sal_num=len(self.files)

    def __getitem__(self, item):

        file=self.files[item]
        names=file.split("/")

        file_path=self.img_root+names[1]

        json_path=self.ref+names[1]
        json_list=os.listdir(json_path)
        jsons_list=[]
        for json_path in json_list:


            if json_path[-3:]=="jpg":
                jsons_list.append(json_path)

        index=random.randint(0,len(jsons_list)-1)

        ref_image=os.path.join(self.ref,names[1],jsons_list[index])
        json_path=ref_image.replace('jpg','json')
        obj_mask,per_mask=comput_mask(img_path=ref_image,json_path=json_path)
        ref_in = load_image(ref_image)
        ref_in,obj_mask,per_mask,_=cv_random_crop_flip_ref(img=ref_in,
                                                         obj_mask=obj_mask,
                                                         per_mask=per_mask,
                                                         resize_size=(self.img_size,self.img_size),
                                                         crop_size=(self.img_size,self.img_size))

        ref_in=Normalize(ref_in)

        obj_mask = torch.Tensor(obj_mask)
        per_mask = torch.Tensor(per_mask)
        ref_in = torch.Tensor(ref_in)

        images=[]
        sub_files=os.listdir(file_path)
        for sub_file in sub_files:
            sub_file_path=file_path+"/"+sub_file
            image=os.listdir(sub_file_path)
            for img in image:
                img_path=sub_file_path+"/"+img

                images.append(img_path)
        random.shuffle(images)

        num_image=len(images)
        input_images = []

        labels = []
        if num_image>=5:
            if num_image==5:
                start=0
            else:
                start=random.randint(0,num_image-6)

            for idx in range(start,start+5):

                img_path=images[idx]

                label_path=img_path.replace("images","masks")
                label_path=label_path.replace("jpg","png")

                tmp_image=load_image(img_path)
                tmp_label=load_label(label_path)

                tmp_image, tmp_label = cv_random_crop_flip(tmp_image,
                                                           tmp_label,
                                                           resize_size=(360,360),
                                                           crop_size=(self.img_size,self.img_size))
                tmp_image=Normalize(tmp_image)

                tmp_image=torch.Tensor(tmp_image)

                tmp_label=torch.Tensor(tmp_label)

                input_images.append(tmp_image)
                labels.append(tmp_label)


        sample = {'image': input_images, 'label': labels,
                  'support_image':ref_in,
                  'obj_mask':obj_mask,'per_mask':per_mask}
        return sample

    def __len__(self):
        # return max(max(self.edge_num, self.sal_num), self.skel_num)
        return self.sal_num


class ImageDataTest(data.Dataset):
    def __init__(self, image_size,root_path,ref_root,txt_path):
        self.root_path=root_path
        self.ref_root=ref_root

        self.test_fold=None
        with open(txt_path,'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]
        self.image_num=len(self.image_list)
        self.image_size=image_size


    def __getitem__(self, item):

        images=[]
        names=[]
        im_sizes=[]
        file_paths=self.image_list[item].split(",")
        #print(file_paths)
        ref_path=os.path.join(self.ref_root,file_paths[0])
        json_path=ref_path.replace(".jpg",".json")
        #print(ref_path)
        obj_mask, per_mask = comput_mask(img_path=ref_path, json_path=json_path)
        ref_img, _ = load_image_test(ref_path, if_resize=True, image_size=self.image_size)

        ref_img=Normalize(ref_img)
        ref_img = torch.Tensor(ref_img)

        obj_mask = cv2.resize(obj_mask, (self.image_size,self.image_size))
        obj_mask = torch.Tensor(obj_mask)
        per_mask = cv2.resize(per_mask, (self.image_size,self.image_size))
        per_mask = torch.Tensor(per_mask)


        file_paths=file_paths[1:]


        for file_path in file_paths:
            image_path=os.path.join(self.root_path,file_path)
            #print(image_path)
            image,im_size=load_image_test(image_path,
                                          image_size=self.image_size,
                                          if_resize=True)
            image=Normalize(image)
            image = torch.Tensor(image)
            images.append(image)
            im_sizes.append(im_size)
            names.append(file_path)


        return {'image': images, 'name': names,
                'size': im_sizes,'support_img':ref_img,
                'obj_mask':obj_mask,'per_mask':per_mask}

    def save_folder(self):
        return self.test_fold

    def __len__(self):
        # return max(max(self.edge_num, self.skel_num), self.sal_num)
        return self.image_num

# get the dataloader (Note: without data augmentation, except saliency with random flip)
def get_loader(batch_size, image_size,img_root,mask_root,ref,lst_path,txts_path,
               mode='train',root_path=None,txt_path=None,num_thread=1,test_ref_root=None):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(image_size=image_size,
                                 img_root=img_root,
                                 mask_root=mask_root,
                                 ref=ref,
                                 lst_path=lst_path,txt_path=txts_path)
    else:
        dataset = ImageDataTest(image_size=image_size,root_path=root_path,ref_root=test_ref_root,txt_path=txt_path)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return data_loader, dataset
