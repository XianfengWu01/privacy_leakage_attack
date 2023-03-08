"""
Example script to run an attack from this repository directly without simulation.

This is a quick example for the imprint module attack described in
- Fowl et al. "Robbing the Fed: Directly Obtaining Private Information in Federated Learning"

All caveats apply. Make sure not to leak any unexpected information.
"""
import sys
import os
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from breaching.attacks.analytic_attack import ImprintAttacker
from breaching.cases.malicious_modifications.imprint import ImprintBlock
from collections import namedtuple
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import h5py
import lpips


def additive_noise(input_gradient, std=0.1):
    """
    Additive noise mechanism for differential privacy
    """
    gradient = [grad + torch.normal(torch.zeros_like(grad), std*torch.ones_like(grad)) for grad in input_gradient]
    return gradient


def gradient_clipping(input_gradient, bound=4):
    """
    Gradient clipping (clip by norm)
    """
    max_norm = float(bound)
    norm_type = 2.0 # np.inf
    device = input_gradient[0].device
    
    if norm_type == np.inf:
        norms = [g.abs().max().to(device) for g in input_gradient]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g, norm_type).to(device) for g in input_gradient]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    
    gradient = [g.mul_(clip_coef_clamped.to(device)) for g in input_gradient]
    return gradient


def gradient_compression(input_gradient, percentage=10):
    """
    Prune by percentage
    """
    device = input_gradient[0].device
    gradient = [None]*len(input_gradient)
    for i in range(len(input_gradient)):
        grad_tensor = input_gradient[i].clone().cpu().numpy()
        flattened_weights = np.abs(grad_tensor.flatten())
        thresh = np.percentile(flattened_weights, percentage)
        grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
        gradient[i] = torch.Tensor(grad_tensor).to(device)
    return gradient


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

    num_users=len(train_pixels)
    users=[i for i in range(num_users)]
    groups=[]
    trainsep = {}
    for i in users:
        trainsep[str(i)]=[train_pixels[i], train_label[i]]

    users=[str(i) for i in users]

    data_aug_phase = cfg.data_aug_phase

    test_set = MyDataset(test_pixels, test_label, data_aug=data_aug_phase, phase='test')

    val_set = MyDataset(val_pixels, val_label, data_aug=data_aug_phase, phase='val')

    return trainsep, test_set, val_set

class data_cfg_default:
    modality = "vision"
    size = (500,)
    classes = 2
    shape = (3, 224, 224)
    normalize = True
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)


class attack_cfg_default:
    type = "analytic"
    attack_type = "imprint-readout"
    label_strategy = None  # Labels are not actually required for this attack
    normalize_gradients = False
    sort_by_bias = False
    text_strategy = "no-preprocessing"
    token_strategy = "decoder-bias"
    token_recovery = "from-limited-embedding"
    breach_reduction = "weight"
    breach_padding = True # Pad with zeros if not enough data is recovered
    impl = namedtuple("impl", ["dtype", "mixed_precision", "JIT"])("float", False, "")


dm = torch.as_tensor(data_cfg_default.mean)[:, None, None]
ds = torch.as_tensor(data_cfg_default.std)[:, None, None]


class default(object):
    def __init__(self, data_path, data_aug_phase=True):
        super().__init__()
        self.data_path = data_path
        self.data_aug_phase = data_aug_phase


def plot(tensor, index, classify):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    if tensor.shape[0] == 1:
        return plt.imshow(tensor[0].permute(1, 2, 0))
    else:
        fig, axes = plt.subplots(8, int(tensor.shape[0] / 8), figsize=(12, 12), subplot_kw={'xticks': [], 'yticks': []})
        
        for i, im in enumerate(tensor):
            x = int(i / 8)
            y = int(i % 8)
            axes[x, y].imshow(im.permute(1, 2, 0))
        if classify == 'original':
            fig.savefig('ODIR_64/original{}.png'.format(index))
        elif classify == 'reconstruct':
            fig.savefig('ODIR_64/reconstruct{}.png'.format(index))


def psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """Standard PSNR."""
    def get_psnr(img_in, img_ref):
        mse = ((img_in - img_ref)**2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(factor**2 / mse))
        elif not torch.isfinite(mse):
            return img_batch.new_tensor(float('nan'))
        else:
            return img_batch.new_tensor(float('inf'))

    if batched:
        psnr = get_psnr(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()

    return psnr.item()


def main():
    # initial
    lpips_score = []
    lpips_score_a = []
    PSNR = []
    RMSE = []
    image_index = 1
    batch_size = 1
    setup = dict(device=torch.device("cpu"), dtype=torch.float)

    # This could be any model:
    # model = torchvision.models.resnet18()
    # from models.modeling import VisionTransformer, CONFIGS

    # model = VisionTransformer(CONFIGS["ViT-B_16"], 224, zero_head=True, num_classes=2, task_num_classes=2)
    model = torch.load('checkpoint/Retina_best_model.pkl', map_location=torch.device("cpu"))
    # model = torch.load('checkpoint/ODIR_best_model.pkl', map_location=torch.device("cpu"))
    # model.load_from(np.load("checkpoint/ViT-B_16.npz"))
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    # It will be modified maliciously:
    block = ImprintBlock(data_cfg_default.shape, num_bins=128)
    model = torch.nn.Sequential(block, model)
    secret = dict(weight_idx=0, bias_idx=1, shape=tuple(data_cfg_default.shape), structure=block.structure)
    secrets = {"ImprintBlock": secret}

    # And your dataset:
    # dataset = torchvision.datasets.ImageNet(root="~/data/imagenet", split="val", transform=transforms)
    cfg = default(data_path='/home/beckham/code/Stanford_HKU/fedavgmodels/Retina')
    _, test_dataset, _ = get_dataset(cfg)
    TestDataLoader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    # from torch.autograd import Variable
    from tqdm import tqdm
    loop = tqdm(enumerate(TestDataLoader), total=len(TestDataLoader))
    for step, (image, labels) in loop:

        datapoint = image
        labels = labels

        # This is the attacker:
        attacker = ImprintAttacker(model, loss_fn, attack_cfg_default, setup)

        # Simulate an attacked FL protocol
        # Server-side computation:
        server_payload = [
            dict(
                parameters=[p for p in model.parameters()], buffers=[b for b in model.buffers()], metadata=data_cfg_default
            )
        ]
        # User-side computation:
        loss = loss_fn(model(datapoint)[0], labels)
        shared_data = [
            dict(
                gradients=additive_noise(torch.autograd.grad(loss, model.parameters())), # defense
                buffers=None,
                metadata=dict(num_data_points=batch_size, labels=labels, local_hyperparams=None,),
            )
        ]

        # Attack:
        reconstructed_user_data, stats = attacker.reconstruct(server_payload, shared_data, secrets, dryrun=False)

        # Do some processing of your choice here. Maybe save the output image?
        
        x_res = reconstructed_user_data['data']
        lpips_loss = lpips.LPIPS(net='vgg', spatial=False)
        lpips_loss_a = lpips.LPIPS(net='alex', spatial=False)
        # PSNR = psnr(img_batch=x_res, ref_batch=datapoint, batched=False)
        # RMSE = (x_res - datapoint).pow(2).mean().item()
        # print(psnr(img_batch=x_res, ref_batch=datapoint, batched=False))
        PSNR.append(psnr(img_batch=x_res, ref_batch=datapoint, batched=True))
        RMSE.append((x_res - datapoint).pow(2).mean().item()) 

        with torch.no_grad():
            # lpips_score = lpips_loss(x_res, datapoint).squeeze().item()
            # lpips_score_a = lpips_loss_a(x_res, datapoint).squeeze().item()
            # print(lpips_loss(x_res, datapoint))
            lpips_score.append(lpips_loss(x_res, datapoint).mean().item())
            lpips_score_a.append(lpips_loss_a(x_res, datapoint).mean().item())

        if batch_size == 1: # Batch size = 1
            original_img = datapoint.mul_(ds).add_(dm).clamp_(0, 1).mul_(255).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].numpy()
            res_img = x_res.mul_(ds).add_(dm).clamp_(0, 1).mul_(255).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].numpy()
            plt.imsave('Retina_{}/original{}.png'.format(batch_size, image_index), original_img)
            plt.imsave('Retina_{}/reconstruct{}.png'.format(batch_size, image_index), res_img)
        elif batch_size > 1: # Batch size = 32, 64 > 1
            plot(datapoint, image_index, 'original')
            plot(x_res, image_index, 'reconstruct')
        image_index = image_index + 1
    lpips_score = np.array(lpips_score)
    lpips_score_a = np.array(lpips_score_a)
    lpips_score = np.array(lpips_score)
    lpips_score_a = np.array(lpips_score_a)
    PSNR = np.array(PSNR)
    RMSE = np.array(RMSE)
    np.savetxt('lpips_score_{}.csv'.format(batch_size), lpips_score, delimiter = ',')
    np.savetxt('lpips_score_a_{}.csv'.format(batch_size), lpips_score_a, delimiter = ',')
    np.savetxt('PSNR_{}.csv'.format(batch_size), PSNR, delimiter = ',')
    np.savetxt('RMSE_{}.csv'.format(batch_size), RMSE, delimiter = ',')
    print('LPIPS score (VGG): {:.3f}, LPIPS score (ALEX): {:.3f}, PSNR: {:.3f}, RMSE: {:.5f}'.format(lpips_score.mean(), lpips_score_a.mean(), PSNR.mean(), RMSE.mean()))


if __name__ == "__main__":
    main()
