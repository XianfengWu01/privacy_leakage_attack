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
from ODIR_dataset import ODIRDataset


def calculate_acuracy_mode_one(model_pred, labels):

    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    pred_one_num = torch.sum(pred_result)
    if pred_one_num == 0:
        return 0, 0
    target_one_num = torch.sum(labels)
    true_predict_num = torch.sum(pred_result * labels)

    precision = true_predict_num / pred_one_num
    recall = true_predict_num / target_one_num
 
    return precision.item(), recall.item()


def main():
    # initial
    epoch_num = 20
    learning_rate = 1e-4
    batch_size = 32
    betas = (0.9, 0.999)
    eps = 1e-08

    # This could be any model:
    # model = torchvision.models.resnet18()
    from models.modeling import VisionTransformer, CONFIGS

    model = VisionTransformer(CONFIGS["ViT-B_16"], 224, zero_head=True, num_classes=8, task_num_classes=8)
    # model.load_from(np.load("checkpoint/ViT-B_16.npz"))
    if torch.cuda.is_available():	
        model.cuda()
        print("cuda is available, and the calculation will be moved to GPU\n")
    else:
        print("cuda is unavailable!")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=0.1)
    loss_fn = torch.nn.BCELoss()
    sigmoid = torch.nn.Sigmoid()

    # And your dataset:
    path = '/home/beckham/code/Stanford_HKU/data/'
    train_set = ODIRDataset(path=path, phase='train')
    # test_dataset = ODIRDataset(path=path, phase='test')
    TrainLoader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    # TestLoader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
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
            loss = loss_fn(sigmoid(output), label) 

            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
            # _, correct_label = torch.max(output, 1)  

            # correct_num = (correct_label == label).sum()

            # trainAcc = correct_num.item() / float(len(label))
            _, recall = calculate_acuracy_mode_one(sigmoid(output), label)

            loop.set_description(f'Epoch [{i + 1}/{epoch_num}]')
            loop.set_postfix(loss = loss.item(), acc = recall)
        
        # testcor = 0
        # model.eval()
        # with torch.no_grad():
        #     for _, (x, y) in enumerate(TestLoader):
        #         # gives batch data, normalize x when iterate train_loader
        #         if torch.cuda.is_available():
        #             batch_x = Variable(x).cuda()
        #             batch_y = Variable(y).cuda()
        #         else:
        #             batch_x = Variable(x)
        #             batch_y = Variable(y)

        #         output = model(batch_x)[0]
        #         loss = loss_fn(sigmoid(output), batch_y)   
        #         _, correct_label = torch.max(output, 1)   
        #         # print('label', correct_label)
        #         # correct_num = (correct_label == batch_y).sum()
        #         # testcor += correct_num.item()
        #         _, recall = calculate_acuracy_mode_one(sigmoid(output), batch_y)
        #         testcor += recall

        # testAcc = testcor / float(len(TestLoader))

        # print('----------------test: Epoch [%d/%d] Acc: %.4f loss: %.4f' % (i + 1, epoch_num, testAcc, loss))
        # # 最后保存模型
        # if BestAcc < testAcc:
        #     BestAcc = testAcc
        torch.save(model, 'ODIR_best_model.pkl')


if __name__ == "__main__":
    main()
