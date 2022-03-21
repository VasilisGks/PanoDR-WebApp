from torch.optim import lr_scheduler
from .layer import init_weights
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from .basemodel import BaseModel
from .basenet import BaseNet
from .PanoPaint_networks import *
from .basemodel import BaseModel
from .basenet import BaseNet
from .GatedConv.network_module import *
import streamlit as st
from helpers import layoutViz

class InpaintingModel(BaseModel):
    def __init__(self, act=F.elu, opt=None, device=None):
        super(InpaintingModel, self).__init__()
        self.opt = opt
        self.init(opt)
        self.device = device
        self.netG = GatedGenerator(self.opt, self.device).to(self.device)
        if self.opt.model_type == 'PanoPaint':
            self.netG = self.load_networks(self.netG, self.opt.eval_chkpnt_folder, self.device)
        elif self.opt.model_type == 'PanoDR':
            self.netG = self.load_networks(self.netG, self.opt.eval_chkpnt_folder_dr, self.device)

        self.model_names = ['D', 'G']
        if self.opt.phase == 'test':
            return

    @st.cache(allow_output_mutation=True, ttl=3600, max_entries=1)
    def load_networks(self, model, load_path, device):
        checkpoint = torch.load(load_path, map_location=torch.device(self.device))
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        model.load_state_dict(checkpoint, strict=True)
        model.to(device)
        return model

    def initData(self, data, epoch, iteration):

        if self.opt.use_irregular_masks:
            self.mask = data["mask"].to(self.device)    #0 in masked area  #s3d
            self.inverse_mask = 1 - data["mask"].to(self.device)   #1 in masked area
            self.foreground_full = None
        else:
            self.inverse_mask = data["foreground"].to(self.device)   #1 in masked area for #m3d
            self.foreground_full = data["foreground_full"].to(self.device) #Full foreground for the input image
            self.mask_fg = (1 - self.foreground_full).to(self.device)
            self.mask = (1 - self.inverse_mask).to(self.device)

        self.f_name = data["img_path"][0]
        self.images = data["img"].to(self.device)
        if self.opt.dataset == 's3d':
            self.gt_label_one_hot = data['label_one_hot'].to(self.device)
        else:
            self.gt_label_one_hot = data['label_one_hot'].to(self.device)
            self.gt_label_one_hot_viz = data['label_viz'].to(self.device)
        self.gt_empty = data["img_gt"].to(self.device)
        
        self.img_path = data["img_path"]
        self.sem_layout = data["label_semantic"].to(self.device)
        self.masked_input = self.images * self.mask + self.inverse_mask
        self.mask_patch_gt = self.gt_empty * self.inverse_mask 
        self.epoch = epoch
        self.iteration = iteration

    def inference_file_inpaint(self, images, mask):
        
        self.f_name = None
        self.images = images
        self.inverse_mask = mask
        self.mask = (1.0-self.inverse_mask)
        self.gt_empty = self.images
        self.masked_input = (self.images * self.mask) + self.inverse_mask
        self.foreground_full = None
        
        self.out, self.structure_model_output, self.gt_layout_3_classes, self.style_codes_gt = self.netG(self.images, 
        self.masked_input, self.inverse_mask, self.foreground_full, self.device, self.opt.use_sean)
        ret =  self.out * self.inverse_mask + (self.images * self.mask)
        ret_masked = self.out * self.inverse_mask
        ret = ret.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 
        ret_masked = ret_masked.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 
        gt_img_masked = self.gt_empty * self.inverse_mask 
        gt_img_masked = gt_img_masked.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 
        masked_input_np = self.masked_input.squeeze_(0).permute(1,2,0).cpu().detach().numpy()
        self.out = self.out.squeeze_(0).permute(1,2,0).cpu().detach().numpy()

        dense_layout = self.structure_model_output.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 
        dense_layout_viz = layoutViz(dense_layout, height=self.images.shape[2], width=self.images.shape[3])
        
        return self.out, ret, masked_input_np, dense_layout_viz