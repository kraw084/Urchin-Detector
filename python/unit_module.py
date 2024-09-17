import torch
import matplotlib.pyplot as plt
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
        #min_values = tmap.amin(dim=(0, 2, 3)).view(1, 3, 1, 1)
        #max_values = tmap.amax(dim=(0, 2, 3)).view(1, 3, 1, 1)
        #tmap = (tmap - min_values)/(max_values - min_values)
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

        new_images = images - (1 - t_maps) * atmospheric_lighting.view((atmospheric_lighting.shape[0], 3, 1, 1))
        return new_images


    def __call__(self, images):
        return self.enhance_images(images)
    

def unit_mod_train_step(unit_mod, opt, images, enhanced_images, detector_Loss, alpha = 0.9, w1=500, w2=0.01, w3=0.01, w4=0.1):
    opt.zero_grad()

    t_maps = unit_mod.last_tmaps
    atmo_map = unit_mod.last_atmo

    #compute degraged and enhanced degraded images
    degraded_images = images * alpha + (1 - alpha) * atmo_map.view((atmo_map.shape[0], 3, 1, 1))
    with torch.no_grad():
        enhanced_degraded_images = unit_mod(degraded_images)
    t_maps_degraded = unit_mod.last_tmaps

   
    #compute losses
    tmap_loss = torch.sum(((alpha * t_maps) - t_maps_degraded)**2)
    tmap_loss /= torch.numel(t_maps)

    #tmap_loss_max = torch.max(tmap_loss)
    #tmap_loss /= tmap_loss_max
    #tmap_loss = torch.sum(tmap_loss) / torch.numel(tmap_loss)

    #ones = torch.ones_like(enhanced_images)
    #zeros = torch.zeros_like(enhanced_images)

    #sp_loss = torch.sum(torch.maximum(enhanced_images, ones) + torch.sum(torch.maximum(enhanced_degraded_images, ones))) -\
    #          torch.sum(torch.minimum(enhanced_images, zeros) + torch.sum(torch.minimum(enhanced_degraded_images, zeros)))

    #sp_loss = torch.maximum(enhanced_images, ones) + (torch.maximum(enhanced_degraded_images, ones)) -\
    #          torch.minimum(enhanced_images, zeros) + torch.minimum(enhanced_degraded_images, zeros)
    
    #sp_loss_max = torch.max(sp_loss)
    #sp_loss = sp_loss / sp_loss_max
    #sp_loss = torch.sum(sp_loss) / torch.numel(sp_loss)

    #tv_loss_h = torch.sum((enhanced_images[:, :, 1:, :] - enhanced_images[:, :, :-1, :])**2)
    #tv_loss_w = torch.sum((enhanced_images[:, :, :, 1:] - enhanced_images[:, :, :, :-1])**2)
    #tv_loss_h /= enhanced_images.shape[2]
    #tv_loss_w /= enhanced_images.shape[3]

    #tv_loss_w_max = torch.max(tv_loss_w) 
    #tv_loss_w /= tv_loss_w_max
    #tv_loss_w = torch.sum(tv_loss_w) / torch.numel(tv_loss_w)

    #tv_loss_h_max = torch.max(tv_loss_h) 
    #tv_loss_h /= tv_loss_h_max
    #tv_loss_h = torch.sum(tv_loss_h) / torch.numel(tv_loss_h)

    #tv_loss = (tv_loss_h + tv_loss_w) #/2
    
    #cc_loss = torch.sum((torch.mean(enhanced_images[:, 0, :, :]) - torch.mean(enhanced_images[:, 1, :, :]))**2) +\
    #          torch.sum((torch.mean(enhanced_images[:, 1, :, :]) - torch.mean(enhanced_images[:, 2, :, :]))**2) +\
    #          torch.sum((torch.mean(enhanced_images[:, 2, :, :]) - torch.mean(enhanced_images[:, 0, :, :]))**2)
    
    #cc_loss_max = torch.max(cc_loss)
    #cc_loss /= cc_loss_max
    #cc_loss = torch.sum(cc_loss) / torch.numel(cc_loss)

    #print("Before weighting:")
    #print(f"tmap loss: {tmap_loss}")
    #print(f"sat loss: {sp_loss}")
    #print(f"var loss: {tv_loss}")
    #print(f"col cast loss: {cc_loss}")

    #print("\nAfter weighting:")
    #print(f"tmap loss: {w1 * tmap_loss}")
    #print(f"sat loss: {w2 * sp_loss}")
    #print(f"var loss: {w3 * tv_loss}")
    #print(f"col cast loss: {w4 * cc_loss}")

    #print(f"\nDet loss: {detector_Loss}")
    
    total_loss = detector_Loss + w1 * tmap_loss #+ w2  * tv_loss + w4 * cc_loss # + w3 * sp_loss

    #print(f"Total loss: {total_loss}")

    #total_loss.backward()
    #opt.step()

    return total_loss

def save_unit_mod(model, i):
    torch.save(model.tmap_network.state_dict(), "models/unit_module/v1" + f"/Epoch{i}.pt")


def load_unit_module(path, c1, c2, k1, k2):
    unit_mod = UnitModule(c1, c2, k1, k2)
    unit_mod.tmap_network.load_state_dict(torch.load(path, weights_only=True))

    return unit_mod
    

if __name__ == "__main__":
    images = os.listdir("data/images")
    random.seed(42)
    random.shuffle(images)

    for im_name in images:
        test_im = read_image("data/images/" + im_name)
        if test_im.shape[1]%4 != 0 or test_im.shape[2]%4 != 0: continue

        test_im = test_im.view((1, *test_im.shape))/255

        model = UnitModule(32, 32, 9, 9)
        output = model(test_im.float())

        #show transmission map
        plt.imshow(model.last_tmaps[0].permute(1, 2, 0).detach().numpy())
        plt.show()

        #show original and enhanced image
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(test_im[0].permute(1, 2, 0).numpy())
        axes[1].imshow(output[0].permute(1, 2, 0).detach().numpy())
        plt.show()

        #unit_mod_train_step(model, None, test_im, 10)

