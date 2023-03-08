import sys
import os
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm


def load_hf_data(cfg, phase):

    data_path = cfg.data_path
    ## this is pre-processed by Miao
    hf = h5py.File(os.path.join(data_path, 'Retina_' + phase + '.h5'), 'r')
    d = hf['examples']

    pixels = d['0']['pixels']
    label = d['0']['label']
    
    return pixels, label


class MyDataset(Dataset):
    def __init__(self, data, label, data_aug=False, phase = 'train'):
        self.data = data
        self.label = label

        if phase == 'train':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.05)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        self.transform = transform

    def __getitem__(self, index):

        x = self.data[index]
        y = self.label[index]
        ## if x is grayimage, then broadcast to 3 images
        ## ADNI to 3 channels
        if len(x.shape) < 3:
            x = np.stack([x, x, x], 2)

        x = np.asarray(x).astype('uint8')

        img = Image.fromarray(x)
        y = np.asarray(y).astype('int64')

        if self.transform is not None:
            img = self.transform(img)

        return img, y
    
    def __len__(self):
        return len(self.data)

#task: integer between -1 and 19 inclusive, -1 means mortality task, 0-19 means icd9 task
def get_dataset(cfg):

  train_pixels, train_label = load_hf_data(cfg, 'train')
  test_pixels, test_label = load_hf_data(cfg, 'test')
  val_pixels, val_label = load_hf_data(cfg, 'val')

  ## just extract half of the data
  train_pixels = train_pixels[::4, :, :, :]
  train_label = train_label[::4, ]

  test_pixels = test_pixels[::2, :, :, :]
  test_label = test_label[::2,]
  val_pixels = val_pixels[::2, :, :, :]
  val_label = val_label[::2, ]

  data_aug_phase = cfg.data_aug_phase

  train_set = MyDataset(train_pixels, train_label, data_aug=data_aug_phase, phase='train')

  test_set = MyDataset(test_pixels, test_label, data_aug=data_aug_phase, phase='test')

  val_set = MyDataset(val_pixels, val_label, data_aug=data_aug_phase, phase='val')

  return train_set, test_set, val_set


class default(object):
    def __init__(self, data_path, data_aug_phase=True):
        super().__init__()
        self.data_path = data_path
        self.data_aug_phase = data_aug_phase


def main():
    # initial
    epoch_num = 250
    learning_rate = 1e-3
    batch_size = 32
    betas = (0.9, 0.999)
    eps = 1e-08

    # This could be any model:
    # model = torchvision.models.resnet18()
    from models.modeling import VisionTransformer, CONFIGS

    model = VisionTransformer(CONFIGS["ViT-B_16"], 224, zero_head=True, num_classes=2, task_num_classes=2)
    # model.load_from(np.load("checkpoint/ViT-B_16.npz"))
    if torch.cuda.is_available():	
        model.cuda()
        print("cuda is available, and the calculation will be moved to GPU\n")
    else:
        print("cuda is unavailable!")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    # And your dataset:
    cfg = default(data_path='/home/beckham/code/Stanford_HKU/fedavgmodels/Retina') # Add your dataset path
    train_set, test_dataset, _ = get_dataset(cfg)
    TrainLoader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    TestLoader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    BestAcc = 0
    for i in range(epoch_num):
        loop = tqdm(enumerate(TrainLoader), total=len(TrainLoader))
        model.train(True)
        for step, (image, label) in loop:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                image = Variable(image).cuda()
                label = Variable(label).cuda()
            else:
                image = Variable(image)
                label = Variable(label)
            output = model(image)[0]
            
            loss = loss_fn(output.view(-1, 2), label.view(-1)) # clear gradients for this training step

            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
            _, correct_label = torch.max(output, 1)  

            correct_num = (correct_label == label).sum()

            trainAcc = correct_num.item() / float(len(label))

            loop.set_description(f'Epoch [{i + 1}/{epoch_num}]')
            loop.set_postfix(loss = loss.item(), acc = trainAcc)
        
        testcor = 0
        model.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(TestLoader):
                # gives batch data, normalize x when iterate train_loader
                if torch.cuda.is_available():
                    batch_x = Variable(x).cuda()
                    batch_y = Variable(y).cuda()
                else:
                    batch_x = Variable(x)
                    batch_y = Variable(y)

                loss = model(batch_x)[0]
                loss = loss_fn(output.view(-1, 2), label.view(-1))    
                _, correct_label = torch.max(output, 1)   
                # print('label', correct_label)
                correct_num = (correct_label == batch_y).sum()
                testcor += correct_num.item()

        testAcc = testcor / float(len(TestLoader)*batch_size)

        print('----------------test: Epoch [%d/%d] Acc: %.4f loss: %.4f' % (i + 1, epoch_num, testAcc, loss))
        # 最后保存模型
        if BestAcc < testAcc:
            BestAcc = testAcc
            torch.save(model, 'Retina_best_model.pkl')


if __name__ == "__main__":
    main()
