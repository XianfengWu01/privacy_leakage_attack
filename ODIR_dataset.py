import sys
import os
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from pathlib import Path


mean = [0.485, 0.456, 0.406]
std = [0.228, 0.224, 0.225]

Keylabel = [
            ['lens dust', 'low image quality', 'normal fundus'], 
            ['no fundus image', 'mild nonproliferative retinopathy', 'severe proliferative diabetic retinopathy', 'diabetic retinopathy', 'severe nonproliferative retinopathy', 'moderate non proliferative retinopathy', 'proliferative diabetic retinopathy'], 
            ['suspected glaucoma', 'optic disk photographically invisible', 'glaucoma'], 
            ['cataract'], 
            ['wet age-related macular degeneration', 'dry age-related macular degeneration'], 
            ['hypertensive retinopathy'], 
            ['myopic retinopathy', 'pathological myopia', 'myopic maculopathy', 'anterior segment image'], 
            ['macular hole', 'vitreous degeneration', 'central retinal artery occlusion', 'congenital choroidal coloboma', 'chorioretinal atrophy with pigmentation proliferation', 'morning glory syndrome', 'retinal pigmentation', 'optic discitis', 'retinochoroidal coloboma', 'old chorioretinopathy', 'retinal artery macroaneurysm', 'epiretinal membrane', 'pigment epithelium proliferation', 'oval yellow-white atrophy', 'post laser photocoagulation', 'maculopathy', 'idiopathic choroidal neovascularization', 'tessellated fundus', 'old branch retinal vein occlusion', 'silicone oil eye', 'macular epiretinal membrane', 'laser spot', 'pigmentation disorder', 'atrophy', 'low image quality,maculopathy', 'atrophic change', 'drusen', 'asteroid hyalosis', 'suspected retinal vascular sheathing', 'branch retinal artery occlusion', 'retinal vascular sheathing', 'optic disc edema', 'white vessel', 'optic nerve atrophy', 'suspected abnormal color of  optic disc', 'retinitis pigmentosa', 'refractive media opacity', 'branch retinal vein occlusion', 'arteriosclerosis', 'suspected retinitis pigmentosa', 'abnormal pigment ', 'wedge white line change', 'macular coloboma', 'epiretinal membrane over the macula', 'retina fold', 'central retinal vein occlusion', 'depigmentation of the retinal pigment epithelium', 'chorioretinal atrophy', 'vessel tortuosity', 'spotted membranous change', 'wedge-shaped change', 'rhegmatogenous retinal detachment', 'post retinal laser surgery', 'myelinated nerve fibers']
            ]

class ODIRDataset(Dataset):
    def __init__(self, path, phase='train'):
        # self.path = path
        self.df = pd.read_excel(os.path.join(path, 'ODIR-5K_Training_Annotations(Updated)_V2.xlsx'))
        if phase == 'train':
            load_phase = 'Training'
        elif phase == 'test':
            load_phase = 'Testing'
        data_root = Path(os.path.join(path, 'ODIR-5K_{}_Images'.format(load_phase)))
        all_image_paths = list(data_root.glob('*.jpg'))
        self.all_image_paths = [str(path) for path in all_image_paths]
        # self.label_names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

        if phase == 'train':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.05)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ])

        self.transform = transform

    def __getitem__(self, index):

        img = Image.open(self.all_image_paths[index])
        img_name = self.all_image_paths[index].split('/')[-1]
        label = np.zeros(8)
        # label = self.df[self.df['ID'] == img_name][self.label_names].values
        img_direction = img_name.split('_')[-1][:-4]
        if img_direction == 'right':
            label_name =self.df[self.df['Right-Fundus'] == img_name]['Right-Diagnostic Keywords']
        elif img_direction == 'left':
            label_name =self.df[self.df['Left-Fundus'] == img_name]['Left-Diagnostic Keywords']
        # print(img_name)
        label_name = [x for x in label_name.str.split('，')]
        
        label_name = label_name[0]
        for i in range(len(label_name)):
            if label_name[i] in Keylabel[0]:
                label[0] = 1
            elif label_name[i] in Keylabel[1]:
                label[1] = 1
            elif label_name[i] in Keylabel[2]:
                label[2] = 1
            elif label_name[i] in Keylabel[3]:
                label[3] = 1
            elif label_name[i] in Keylabel[4]:
                label[4] = 1
            elif label_name[i] in Keylabel[5]:
                label[5] = 1
            elif label_name[i] in Keylabel[6]:
                label[6] = 1
            elif label_name[i] in Keylabel[7]:
                label[7] = 1
        
        label = torch.FloatTensor(label)

        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.all_image_paths)


if __name__ == "__main__":
    data_path = '/home/beckham/code/Stanford_HKU/data/'
    # df = pd.read_csv(os.path.join(data_path, 'ODIR.csv'))
    # phase = 'train'
    # if phase == 'train':
    #     load_phase = 'Training'
    # elif phase == 'test':
    #     load_phase = 'Testing'
    # data_root = Path(os.path.join(data_path, 'ODIR-5K_{}_Images'.format(load_phase)))
    # all_image_paths = list(data_root.glob('*.jpg'))
    # all_image_paths = [str(path) for path in all_image_paths]
    # label_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']
    # print(all_image_paths)
    Training_Set = ODIRDataset(path=data_path, phase='test')
    TrainLoader = DataLoader(dataset=Training_Set, batch_size=32, shuffle=True, drop_last=True)
    for i in range(250):
        loop = tqdm(enumerate(TrainLoader), total=len(TrainLoader))
        for step, (image, label) in loop:
            if torch.cuda.is_available():
                image = Variable(image).cuda()
                label = Variable(label).cuda()
            else:
                image = Variable(image)
                label = Variable(label)
        

