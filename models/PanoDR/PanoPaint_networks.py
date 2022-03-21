import torch
from .layer import init_weights
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
import functools
from .GatedConv.network_module import *
from .SEAN._normalization import * 
from .SEAN._spade_arch import *
from .Unet.unet_model import UNet as UNet_s3d
from .featSegm import *
from helpers import to_one_hot

class GatedGenerator(nn.Module):
    def __init__(self, opt, device, sc=False):
        super(GatedGenerator, self).__init__()
        self.opt = opt
        self.patch_dim = int(self.opt.width / 8)
        self.style_codes_pred = None
        #Linear semantics 
        if self.opt.use_LS:
            self.ls_out = None
            self.pred_layout = None
            self.dims = [256, 256, 256]
            self.layers = [0, 1, 2]
            self.SE = EXTRACTOR_POOL[self.opt.type_LS](n_class=self.opt.out_classes, dims=self.dims, layers=self.layers)
            self.Temperature = self.opt.Temperature
        else:
            self.structure_model = UNet_s3d(n_channels = opt.in_layout_channels, n_classes = opt.num_classes, bilinear=False)
            self.opt.ignore_id_softmax = False

        if opt.type_sp == 'SEAN':
            self.Zencoder = Zencoder(opt.in_SEAN_channels, opt.style_code_dim, use_soft_sean=self.opt.use_softSean)
            self.spade_block_1 = SPADEResnetBlock(opt.latent_channels*4, opt.latent_channels*4, opt.in_spade_channels, device=device, Block_Name='up_0', use_soft_sean=self.opt.use_softSean)
            self.spade_block_2 = SPADEResnetBlock(opt.latent_channels*2, opt.latent_channels*2, opt.in_spade_channels, device=device, Block_Name='up_1', use_soft_sean=self.opt.use_softSean)
        
        self.refinement1 = nn.Sequential(
            # Surrounding Context Encoder
            GatedConv2d(opt.in_channels, opt.latent_channels, opt.first_kernel_size, 2, 2, pad_type = opt.pad_type, activation = opt.activation, norm='none', sc=sc),
            GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sc=sc),
        )
        self.refinement2 = nn.Sequential(
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sc=sc),
        )
        self.refinement3 = nn.Sequential(
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sc=sc),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sc=sc),
        )
        self.refinement4 = nn.Sequential(
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sc=sc),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sc=sc),
        )
        self.refinement5 = nn.Sequential(
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sc=sc),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sc=sc),
        )
        self.refinement6 = nn.Sequential(
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sc=sc),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sc=sc),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, sc=sc),
        )
        #Structure-Aware Decoder
        self.refine_dec_1 = nn.Sequential(nn.Upsample(scale_factor=2),
        GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, activation = opt.activation, pad_type='zero', norm=opt.norm, sc=sc),
        )
        self.refine_dec_2 =  GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, sc=sc)
        self.refine_dec_3 = nn.Sequential(nn.Upsample(scale_factor=2), 

        GatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type ='zero', activation = opt.activation, sc=sc),
        )
        self.refine_dec_4 = GatedConv2d(opt.latent_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad_type, norm='none', activation = 'tanh', sc=sc)

        self.conv_pl3 = nn.Sequential(
            GatedConv2d(256, 256, 3, 1, 1, activation=opt.activation, norm=opt.norm)
        )
        self.conv_pl3_down = nn.Sequential(
            GatedConv2d(512, 256, 3, 1, 1, activation=opt.activation, norm=opt.norm)
        )
        self.conv_pl2 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 1, activation=opt.activation, norm=opt.norm),
            GatedConv2d(128, 128, 3, 1, 2, dilation=2, activation=opt.activation, norm=opt.norm)
        )
        self.conv_pl2_down = nn.Sequential(
            GatedConv2d(256, 128, 3, 1, 1, activation=opt.activation, norm=opt.norm)
        )
        self.conv_pl1 = nn.Sequential(
            GatedConv2d(32, 32, 3, 1, 1, activation=opt.activation, norm=opt.norm),
            GatedConv2d(32, 32, 3, 1, 2, dilation=2, activation=opt.activation, norm=opt.norm)

        )

    def forward(self, img, masked_input, inverse_mask, foreground, device, use_sean):
        ls_feats = []

        #Enconder + bottleneck
        second_out_att1 = self.refinement1(torch.cat((masked_input, inverse_mask), 1))   #([B, 128, 64, 128])
        second_out = self.refinement2(second_out_att1)
        second_out = self.refinement3(second_out)
        second_out = self.refinement4(second_out)
        if self.opt.use_LS:   
            ls_feats.append(second_out)
        
        second_out_att2 = self.refinement5(second_out)# ([B, 256, 32, 64])3
        if self.opt.use_LS:
            ls_feats.append(second_out_att2)
        
        second_out = self.refinement6(second_out_att2)
        if self.opt.use_LS:
            ls_feats.append(second_out)
        
        #Linear semantics component  
        if self.opt.use_LS:  
            self.ls_out = self.SE(ls_feats, size=(self.opt.height, self.opt.width))
            self.pred_layout = self.ls_out[-1] 
            self.pred_layout = self.pred_layout/self.Temperature
            self.pred_layout = torch.softmax(self.pred_layout, dim=1)
            self.sem_layout_hot = self.pred_layout
       
        else: 
            gt_layout = self.structure_model(masked_input).clone()
            gt_layout = to_one_hot(self.opt, gt_layout, ignore_index=self.opt.ignore_id_softmax, num_classes=self.opt.num_classes)
            self.sem_layout_hot = gt_layout

        if self.opt.use_attention:
            patch_att = self.calculate_patches(self.patch_dim, inverse_mask, inverse_mask.shape[3])
            att = self.compute_attention(second_out, patch_att)
        
        if self.opt.type_sp == 'SEAN':
            style_codes = self.Zencoder(input=img, segmap=self.sem_layout_hot, foreground=foreground, exclude_foreground=True)
            z = self.spade_block_1(second_out, self.sem_layout_hot, style_codes)
            if self.opt.use_attention:
                #concatenate feature vector with attention transfered feature vector
                z=torch.cat((z,self.conv_pl3(self.attention_transfer(z, att))), 1)
                z=self.conv_pl3_down(z)                
            second_out = self.refine_dec_1(z)

            second_out = self.refine_dec_2(second_out)
            z = self.spade_block_2(second_out, self.sem_layout_hot, style_codes)
            if self.opt.use_attention:
                z=torch.cat((z,self.conv_pl2(self.attention_transfer(second_out_att1, att))), 1)
                z=self.conv_pl2_down(z)  
     
            second_out = self.refine_dec_3(z)
            second_out = self.refine_dec_4(second_out)
        else: 
            second_out = self.refine_dec_1(second_out)
            second_out = self.refine_dec_2(second_out)
            second_out = self.refine_dec_3(second_out)
            second_out = self.refine_dec_4(second_out)
        second_out = torch.clamp(second_out, 0, 1)

        return second_out, self.sem_layout_hot, style_codes, self.style_codes_pred

    def calculate_patches(self, patch_num, mask, raw_size):
        pool = nn.MaxPool2d(raw_size // patch_num)
        patch_fb = pool(mask)
        return patch_fb
    
    def extract_image_patches(self, img, patch_num):
        b, c, h, w = img.shape
        img = torch.reshape(img, [b, c, patch_num, h//patch_num, patch_num, w//patch_num])
        img = img.permute([0, 2, 4, 3, 5, 1])
        return img

    def compute_attention(self, feature, patch_att):
        b = feature.shape[0]
        feature = F.interpolate(feature, scale_factor=0.5, mode='bilinear')  # in: [B, C:32, 32, 32]
        p_fb = torch.reshape(patch_att, [b, patch_att.shape[2] * patch_att.shape[3], 1])
        p_matrix = torch.bmm(p_fb, (1 - p_fb).permute([0, 2, 1]))
        f = feature.permute([0, 2, 3, 1]).reshape([b, feature.shape[2] *    feature.shape[3], feature.shape[1]])
        c = self.cosine_Matrix(f, f) * p_matrix
        attention = F.softmax(c, dim=2) * p_matrix
        return attention

    def attention_transfer(self, feature, attention):  # feature: [B, C, H, W]
        b_num, c, h, w = feature.shape
        f = self.extract_image_patches(feature, 32)
        f = torch.reshape(f, [b_num, attention.shape[1], -1])
        f = torch.bmm(attention, f)
        f = torch.reshape(f, [b_num, 32, 32, h // 32, w // 32, c])
        f = f.permute([0, 5, 1, 3, 2, 4])
        f = torch.reshape(f, [b_num, c, h, w])
        return f

    def cosine_Matrix(self, _matrixA, _matrixB):
        _matrixA_matrixB = torch.bmm(_matrixA, _matrixB.permute([0, 2, 1]))
        _matrixA_norm = torch.sqrt((_matrixA * _matrixA).sum(axis=2)).unsqueeze(dim=2)
        _matrixB_norm = torch.sqrt((_matrixB * _matrixB).sum(axis=2)).unsqueeze(dim=2)
        return _matrixA_matrixB / torch.bmm(_matrixA_norm, _matrixB_norm.permute([0, 2, 1]))
