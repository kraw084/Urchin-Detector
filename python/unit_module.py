import torch

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

    def forward(self, x):
        features = self.unit_backbone(x)
        t_map = self.t_head(features)

        return t_map
    

class UnitModule:
    def __init__(self, c1, c2, k1, k2):
        self.tmap_network = TMapGenerator(c1, c2, k1, k2)


    def enhance_images(self, images):
        t_maps = self.tmap_network(images)
        atmospheric_lighting = torch.mean(images, dim=(2, 3))

        new_images = images - (1 - t_maps) * atmospheric_lighting.view((atmospheric_lighting.shape[0], 3, 1, 1))
        return new_images


    def __call__(self, images):
        return self.enhance_images(images)

    
test_im = torch.rand((1, 3, 640, 640))
print("Original shape:", test_im.shape)

model = UnitModule(32, 32, 9, 9)
output = model(test_im)

print(output.shape)

