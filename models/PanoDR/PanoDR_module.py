from torch.optim import lr_scheduler
from .layer import init_weights
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from .basemodel import BaseModel
from .basenet import BaseNet
from .GatedConv.network_module import *
from .PanoDR_networks import *
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

class PanoDR(BaseModel):
    def __init__(self, act=F.elu, opt=None, device=None):
        super(PanoDR, self).__init__()
        self.opt = opt
        self.init(opt)
        self.device = device
        self.netG = GatedGenerator(self.opt, self.device).to(self.device)
        init_weights(self.netG, init_type=self.opt.init_type)
        
        if self.opt.structure_model != "":
            checkpoint = torch.hub.load_state_dict_from_url(opt.segmentation_model_chkpnt, map_location='cpu')
            self.netG.structure_model.load_state_dict(checkpoint)
            self.netG.structure_model.to(self.device)
            print("Freezing Layout segmentation network's weights\n")
            for param in self.netG.structure_model.parameters():
                param.requires_grad = False

        norm_layer = get_norm_layer()

        self.model_names = ['D', 'G']
        if self.opt.phase == 'test':
            return

    def get_current_learning_rate(self):
        return self.optimizers[0].param_groups[0]['lr']

    def update_learning_rate(self):
        for schedular in self.schedulers:
            schedular.step()

    def initData(self, data, epoch, iteration):
        self.mask = data["mask"].to(self.device)
        self.f_name = data["img_path"][0]
        self.images = data["img"].to(self.device)
        self.gt_label_one_hot = data['label_one_hot'].to(self.device)
        self.mask = data["mask"].to(self.device)    #0 in masked area
        self.gt_empty = data["img_gt"].to(self.device)
        self.inverse_mask = 1 - data["mask"].to(self.device)   #1 in masked area
        self.img_path = data["img_path"]
        self.sem_layout = data["label_semantic"].to(self.device)
        self.masked_input = self.images * self.mask + self.inverse_mask
        self.mask_patch_gt = self.gt_empty * self.inverse_mask 
        self.epoch = epoch
        self.iteration = iteration


    def inference_file(self, images, mask, f_name):

        result_path = os.path.join(self.opt.eval_path, "output/")
        os.makedirs(result_path, exist_ok=True)

        self.f_name = None
        self.images = images
        self.inverse_mask = mask
        self.mask = (1.0-self.inverse_mask)
        self.gt_empty = self.images
        masked_input = (self.images * self.mask) + self.inverse_mask
        _, out , self.structure_model_output, self.structure_model_output_soft = self.netG(self.images, self.inverse_mask, masked_input,  self.device, self.opt.use_sean)
        ret =  out * self.inverse_mask + (self.images * self.mask)
        ret_masked = ret * self.inverse_mask
        ret = ret.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 
        raw_ret = out.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 
        ret_masked = ret_masked.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 
        gt_img_masked = self.gt_empty * self.inverse_mask 
        gt_img_masked = gt_img_masked.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 

        masked_input_np = masked_input.squeeze_(0).permute(1,2,0).cpu().detach().numpy()

        _layout = self.structure_model_output_soft.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 
        a=np.argmax(_layout, axis=2)
        z=np.zeros((256,512,3))
        z[a==0] = (255,0,0);z[a==1] = (255,255,255);z[a==2] = (0,0,255)
        z=z.astype(np.float32)
        return raw_ret, ret, masked_input_np, z
        