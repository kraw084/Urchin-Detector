import torch
import matplotlib.pyplot as plt

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
    

def unit_mod_train_step(unit_mod, opt, images, detector_Loss, alpha = 0.9, w1=500, w2=0.01, w3=0.01, w4=0.1):
    enhanced_images = unit_mod(images)
    t_maps = unit_mod.last_tmaps
    #backbone_features = unit_mod.tmap_network.last_features
    atmo_map = unit_mod.last_atmo

    degraded_images = images * alpha + (1 - alpha) * atmo_map.view((atmo_map.shape[0], 3, 1, 1))
    enhanced_degraded_images = unit_mod(degraded_images)
    t_maps_degraded = unit_mod.last_tmaps
    #backbone_features_degraded = unit_mod.tmap_network.last_features
   
    tmap_loss = torch.sum(((alpha * t_maps) - t_maps_degraded)**2)

    sp_loss = torch.sum(torch.clip(enhanced_images, max=1) + torch.sum(torch.clip(enhanced_degraded_images, max=1))) -\
              torch.sum(torch.clip(enhanced_images, min=0) + torch.sum(torch.clip(enhanced_degraded_images, min=0)))

    tv_loss = torch.sum((enhanced_images[:, :, 1:, :] - enhanced_images[:, :, :-1, :])**2) +\
              torch.sum((enhanced_images[:, :, :, 1:] - enhanced_images[:, :, :, :-1])**2)
    
    cc_loss = torch.sum((torch.mean(enhanced_images[:, 0, :, :]) - torch.mean(enhanced_images[:, 1, :, :]))**2) +\
              torch.sum((torch.mean(enhanced_images[:, 1, :, :]) - torch.mean(enhanced_images[:, 2, :, :]))**2) +\
              torch.sum((torch.mean(enhanced_images[:, 2, :, :]) - torch.mean(enhanced_images[:, 0, :, :]))**2)
    
    total_loss = detector_Loss + w1 * tmap_loss + w2 * sp_loss + w3 * tv_loss + w4 ** cc_loss

    total_loss.backward()
    opt.step()
    opt.zero_grad()
    
if __name__ == "__main__":
    test_im = torch.randint(0, 256, (1, 3, 640, 640), dtype=torch.int32)
    model = UnitModule(32, 32, 9, 9)

    output = model(test_im.float())
    print(torch.min(output))
    print(torch.max(output))


    #unit_mod_train_step(model, None, test_im, 10)

