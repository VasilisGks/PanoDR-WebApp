import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from ._normalization import SPADE, ACE


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, input_nc, status = None, device = None, Block_Name=None, use_rgb=True, norm_G=False, use_soft_sean=False):
        super().__init__()

        self.use_rgb = use_rgb
        self.use_soft_sean = use_soft_sean
        self.Block_Name = Block_Name
        self.status = status
        self.device = device

        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        #spade_config_str = norm_G.replace('spectral', '')


        ###########  Modifications 1
        normtype_list = ['spadeinstance3x3', 'spadesyncbatch3x3', 'spadebatch3x3']
        our_norm_type = 'spadeinstance3x3'

        self.ace_0 = ACE(our_norm_type, fin, 3, self.device, ACE_Name= Block_Name + '_ACE_0', status=self.status, spade_params=[fin, input_nc], use_rgb=use_rgb, use_soft_sean=self.use_soft_sean)
        ###########  Modifications 1


        ###########  Modifications 1
        self.ace_1 = ACE(our_norm_type, fmiddle, 3, self.device, ACE_Name= Block_Name + '_ACE_1', status=self.status, spade_params=[fmiddle, input_nc], use_rgb=use_rgb, use_soft_sean=self.use_soft_sean)
        ###########  Modifications 1

        if self.learned_shortcut:
            self.ace_s = ACE(our_norm_type, fin, 3, self.device, ACE_Name= Block_Name + '_ACE_s', status=self.status, spade_params=[fin, input_nc], use_rgb=use_rgb)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, style_codes, obj_dic=None):


        x_s = self.shortcut(x, seg, style_codes, obj_dic)


        ###########  Modifications 1
        dx = self.ace_0(x, seg, style_codes, obj_dic)

        dx = self.conv_0(self.actvn(dx))

        dx = self.ace_1(dx, seg, style_codes, obj_dic)

        dx = self.conv_1(self.actvn(dx))
        ###########  Modifications 1


        out = x_s + dx
        return out

    def shortcut(self, x, seg, style_codes, obj_dic):
        if self.learned_shortcut:
            x_s = self.ace_s(x, seg, style_codes, obj_dic)
            x_s = self.conv_s(x_s)

        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


class Zencoder(torch.nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=2, norm_layer=nn.InstanceNorm2d, use_soft_sean=False):
        super(Zencoder, self).__init__()
        self.output_nc = output_nc
        self.use_soft_sean = use_soft_sean
        #self.exclude_foreground = exclude_foreground
        self.background_mask = None

        model = [nn.ReflectionPad2d(1), nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0),
                 norm_layer(ngf), nn.LeakyReLU(0.2, False)]
        
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, False)]

        for i in range(1):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.LeakyReLU(0.2, False)]

        model += [nn.ReflectionPad2d(1), nn.Conv2d(256, output_nc, kernel_size=3, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)


    def forward(self, input, segmap, foreground, is_pred = False, exclude_foreground = False):
        #Obtain background mask
        if foreground is not None:
            self.background_mask = 1 - foreground
            segmap = segmap * self.background_mask
        
        codes = self.model(input)
        # if is_pred == True:
        #     return codes
        segmap = F.interpolate(segmap, size=codes.size()[2:], mode='nearest')

        b_size = codes.shape[0]
        f_size = codes.shape[1]
        s_size = segmap.shape[1]

        codes_vector = torch.zeros((b_size, s_size, f_size), dtype=codes.dtype, device=codes.device)

        for i in range(b_size):
            for j in range(s_size):
                component_mask_area = torch.sum(segmap.bool()[i, j])
                if component_mask_area > 0:
                    alphas_codes_component = segmap[i,j].masked_select(segmap.bool()[i, j])
                    codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]).reshape(f_size,  component_mask_area)

                    if self.use_soft_sean == True:
                        codes_component_feature = (torch.mul(codes_component_feature, alphas_codes_component)) 
                        codes_component_feature = torch.sum(codes_component_feature, dim=1) / torch.sum(alphas_codes_component)
                        codes_vector[i][j] = codes_component_feature
                    else:
                        codes_component_feature = (torch.mul(codes_component_feature, alphas_codes_component)).mean(1) 

                    codes_vector[i][j] = codes_component_feature

        return codes_vector