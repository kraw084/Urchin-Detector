import torch
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data
from torchvision.io import read_image
import os
import random

class LKBlock(torch.nn.Module):
    def __init__(self, c, k):
        super().__init__()

        self.first_layer = torch.nn.Conv2d(c, c, 1)

        self.branch_1 = torch.nn.Sequential(
            torch.nn.Conv2d(c, c, k, groups=c, padding=(k-1)//2),
            torch.nn.GroupNorm(8, c)
        )

        self.branch_2 = torch.nn.Sequential(
            torch.nn.Conv2d(c, c, 3, groups=c, padding=1),
            torch.nn.GroupNorm(8, c),
            torch.nn.ReLU()
        )

        self.last_layer = torch.nn.Conv2d(c, c, 1)
        self.last_norm = torch.nn.GroupNorm(8, c)

    def forward(self, x):
        x_1 = self.first_layer(x)

        x_2 = self.branch_1(x_1)
        x_3 = self.branch_2(x_1)

        x_4 = torch.add(x_2, x_3)
        x_5 = self.last_layer(x_4)
        x_6 = torch.add(x, x_5)
        x_7 = self.last_norm(x_6)
        return x_7


class TMapGenerator(torch.nn.Module):
    def __init__(self, c1, c2, k1, k2):
        super().__init__()
        
        self.unit_backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, c1, 3, 2, 1),
            torch.nn.Conv2d(c1, c2, 3, 2, 1),
            LKBlock(c2, k1),
            LKBlock(c2, k2)
        )
        
        self.t_head = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(c2, c2, 3, padding=1),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(c2, 3, 3, padding=1),
            torch.nn.Sigmoid()
        )

        self.last_features = None

    def forward(self, x):
        features = self.unit_backbone(x)
        self.last_features = features

        t_map = self.t_head(features)
        return t_map
    

class BasicTMapGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(torch.nn.Conv2d(3, 9, 3, 2, 1),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(9, 9, 3, 2, 1),
                                          torch.nn.ReLU(),
                                          
                                          torch.nn.Upsample(scale_factor=2),
                                          torch.nn.Conv2d(9, 9, 3, padding=1),
                                          torch.nn.Upsample(scale_factor=2),
                                          torch.nn.Conv2d(9, 3, 3, padding=1))
        
    def forward(self, x):
        tmap = self.layers(x)
        min_values = tmap.amin(dim=(0, 2, 3)).view(1, 3, 1, 1)
        max_values = tmap.amax(dim=(0, 2, 3)).view(1, 3, 1, 1)
        tmap = (tmap - min_values)/(max_values - min_values)
        return tmap    


class UnitModule:
    def __init__(self, c1, c2, k1, k2):
        self.tmap_network = TMapGenerator(c1, c2, k1, k2)

        self.last_tmaps = None
        self.last_atmo = None


    def enhance_images(self, images):
        t_maps = self.tmap_network(images)
        self.last_tmaps = t_maps

        atmospheric_lighting = torch.mean(images, dim=(2, 3))
        self.last_atmo = atmospheric_lighting

        new_images = (images - (1 - t_maps) * atmospheric_lighting.view((atmospheric_lighting.shape[0], 3, 1, 1))) / (t_maps + 1e-9)
        return new_images


    def __call__(self, images):
        return self.enhance_images(images)
    

def min_max_scale(values):
    min_val = torch.min(values)
    max_val = torch.max(values)
    values = (values - min_val) / (max_val - min_val)
    val = torch.sum(values) / torch.numel(values)
    return val


def calc_tmap_loss(tmap, tmap_deg, alpha):
    tmap_loss = ((alpha * tmap) - tmap_deg)**2
    size = (tmap_loss.shape[2] * tmap_loss.shape[3])
    tmap_loss = (torch.sum(tmap_loss, dim=(1, 2, 3)))
    tmap_loss = tmap_loss / size
    return torch.mean(tmap_loss)


def calc_sp_loss(enh_ims, enh_deg_ims):
    ones = torch.ones_like(enh_ims)
    zeros = torch.zeros_like(enh_ims)
    sp_loss = (torch.maximum(enh_ims, ones) + torch.maximum(enh_deg_ims, ones)) -\
              (torch.minimum(enh_ims, zeros) + torch.minimum(enh_deg_ims, zeros))

    sp_loss = torch.sum(sp_loss, dim=(1, 2, 3))
    size = (enh_ims.shape[2] * enh_ims.shape[3])
    sp_loss = sp_loss / size
    return torch.mean(sp_loss)


def calc_tv_loss(enh_ims):
    tv_loss_h = (enh_ims[:, :, 1:, :] - enh_ims[:, :, :-1, :])**2
    tv_loss_w = (enh_ims[:, :, :, 1:] - enh_ims[:, :, :, :-1])**2
    
    tv_loss_h = torch.sum(tv_loss_h, dim=(1, 2, 3)) / (enh_ims.shape[2] * enh_ims.shape[3])
    tv_loss_w = torch.sum(tv_loss_w, dim=(1, 2, 3)) / (enh_ims.shape[2] * enh_ims.shape[3])
    tv_loss = tv_loss_w + tv_loss_h
    return torch.mean(tv_loss)
    

def calc_cc_loss(enh_ims):
    r_mean = torch.mean(enh_ims[:, 0, :, :], dim=(1, 2))
    g_mean = torch.mean(enh_ims[:, 1, :, :], dim=(1, 2))
    b_mean = torch.mean(enh_ims[:, 2, :, :], dim=(1, 2))

    rg_loss = (r_mean - g_mean)**2
    gb_loss = (g_mean - b_mean)**2
    br_loss = (b_mean - r_mean)**2

    cc_loss = rg_loss + gb_loss + br_loss
    return torch.mean(cc_loss)

def unit_mod_train_step(unit_mod, opt, images, enhanced_images, detector_Loss, alpha = 0.9, w1=500, w2=0.01, w3=0.01, w4=0.1, debug=False):
    if not opt is None:
        opt.zero_grad()

    t_maps = unit_mod.last_tmaps
    atmo_map = unit_mod.last_atmo

    #compute degraged and enhanced degraded images
    degraded_images = images * alpha + (1 - alpha) * atmo_map.view((atmo_map.shape[0], 3, 1, 1))
    with torch.no_grad():
        enhanced_degraded_images = unit_mod(degraded_images)
    t_maps_degraded = unit_mod.last_tmaps

    #compute losses
    tmap_loss = calc_tmap_loss(t_maps, t_maps_degraded, alpha)
    sp_loss = calc_sp_loss(enhanced_images, enhanced_degraded_images)
    tv_loss = calc_tv_loss(enhanced_images)
    cc_loss = calc_cc_loss(enhanced_images)
    
    total_loss = detector_Loss + w1 * tmap_loss + w2 * sp_loss + w3 * tv_loss + w4 * cc_loss

    if debug:
        print("Before weighting:")
        print(f"tmap loss: {tmap_loss}")
        print(f"sat loss: {sp_loss}")
        print(f"var loss: {tv_loss}")
        print(f"col cast loss: {cc_loss}")

        print("\nAfter weighting:")
        print(f"tmap loss: {w1 * tmap_loss}")
        print(f"sat loss: {w2 * sp_loss}")
        print(f"var loss: {w3 * tv_loss}")
        print(f"col cast loss: {w4 * cc_loss}")

        print(f"\nDet loss: {detector_Loss}")
        print(f"Total loss: {total_loss}")
    

    return total_loss

    
def save_unit_mod(model, i):
    torch.save(model.tmap_network.state_dict(), "models/unit_module/v5" + f"/Epoch{i}.pt")


def load_unit_module(path, c1, c2, k1, k2):
    unit_mod = UnitModule(c1, c2, k1, k2)
    unit_mod.tmap_network.load_state_dict(torch.load(path, weights_only=True))

    return unit_mod
    


images = os.listdir("data/images")
random.shuffle(images)
images = images[:1000]

import torch
import torchvision
from tqdm import tqdm

model = UnitModule(32, 32, 9, 9)

opt = torch.optim.Adam(model.tmap_network.parameters(), 0.001)

class ds(torch.utils.data.Dataset):

    def __len__(self):
        return len(images)
    
    def __getitem__(self, index):
        im_name = images[index]
        test_im = read_image("data/images/" + im_name)
        test_im = torchvision.transforms.Resize((640,640))(test_im)
        test_im = test_im/255

        return test_im.float()

dataset = ds()
dl = torch.utils.data.DataLoader(dataset, 32, num_workers=0)

for i in range(100):
    avg_loss = 0
    count = 0
    for b in tqdm(dl):
        opt.zero_grad()

        output = model(b)

        loss = unit_mod_train_step(model, opt, b, output, 0, 0.9, 500, 0.01, 0, 0, debug=False)

        loss.backward()
        opt.step()

        avg_loss += loss.item()
        count += 1

    print(f"Epoch {i}: loss {avg_loss/count}")
