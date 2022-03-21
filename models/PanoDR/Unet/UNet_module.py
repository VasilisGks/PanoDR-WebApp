from torch.optim import lr_scheduler
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from vcl3datlantis.dataloaders.label_mapping import class_mapping, vizLayout
from vcl3datlantis.models.Unet.unet_model import *
from vcl3datlantis.models.Unet.init import *
from vcl3datlantis.models.Unet.unet_parts import *
from vcl3datlantis.models.Unet.utils.framework import *
from vcl3datlantis.losses.seg_losses import *
from vcl3datlantis.models.PanoDR.basemodel import *
from vcl3datlantis.metrics.iou import *
from vcl3datlantis.method.helpers import to_one_hot

class ModelModule(BaseModel):
    def __init__(self, opt, device=None):
        super(ModelModule, self).__init__()
        self.init(opt)
        self.opt = opt
        self.device = device
        if self.opt.model_type == 'unet':
            self.netunet = UNet(self.opt.input_nc, self.opt.num_classes, False) 
            
        initialize_weights(self.netunet, self.opt.weight_init, pred_bias=self.opt.pred_bias)
        self.netunet = self.netunet.to(device) 

        if self.opt.phase == 'eval':
            self.netunet.eval()
            return

        self.netunet.train()
        self.optimizer = init_optimizer(self.netunet, self.opt)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [self.opt.milestone_1, self.opt.milestone_2], self.opt.lr_gamma)
        self.loss = getSegLoss(self.opt.seg_loss_type, num_classes=self.opt.num_classes, ignore_class=self.opt.ignore_class)

        self.model_names = []
        self.model_names.append(self.opt.model_type)

        self.pred = None
        self.ce_loss = None
        self.viz_loss = None
        self.pred_viz = None
        self.pred_one_hot = None
        self.gt_viz = None
        #Metric
        self.metric_iou = IoU()

    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']
    
    def update_learning_rate(self):
        self.scheduler.step()
    
    def initData(self, data, epoch, iteration):
        self.img = data["img"].to(self.device)
        if self.opt.w_foreground == True:
            self.target_label = data['foreground'].to(self.device)
            self.gt_viz = data["foreground_viz"].to(self.device).squeeze_(0)
            self.target_one_hot = data["foreground_one_hot"].to(self.device)
        else: 
            self.target_label = data['label_semantic'].to(self.device)
            self.gt_viz = data["label_viz"].to(self.device).squeeze_(0)
            self.target_one_hot = data["label_one_hot"].to(self.device)

        self.foreground = data['foreground'].to(self.device)
        self.f_name = data["img_path"][0]
        self.target_label_viz = data['label_viz'].to(self.device)
        self.foreground_viz = data['foreground_viz'].to(self.device)
        self.epoch = epoch
        self.iteration = iteration

    def forward(self):
        self.pred = torch.softmax(self.pred, dim=1)
        #self.ce_loss = self.opt.lambda_bce * self.loss(self.pred, self.target_label.squeeze_(0).long())
        self.ce_loss = self.opt.lambda_bce * self.loss(self.pred, self.target_label.squeeze_(0).squeeze_(1).long())

    def backward(self):
        self.ce_loss.backward()

    def optimize_parameters(self):
        self.pred = self.netunet(self.img)
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()
        #self.update_learning_rate()
        
    def get_current_losses(self):
        self.vis_loss = {}
        self.vis_loss.update({'Cross_entropy': self.ce_loss.detach()})
        return self.vis_loss

    def get_current_visuals(self):
        if self.opt.w_foreground == True:
            self.pred_viz = torch.argmax(self.pred, dim=1, keepdim=True)
        else:
            self.pred_viz = torch.argmax(self.pred, dim=1, keepdim=True)
            self.pred_viz = vizLayout(self.opt.num_classes, self.img.shape, self.pred_viz, self.device)

        vis_dict = {
        'Image': self.img.cpu().detach(),
        'Prediction': self.pred_viz.cpu().detach(),
        'Ground_Truth': self.gt_viz.cpu().detach().unsqueeze(0)
        } 
        return vis_dict
    
    def evaluate(self):
        self.pred_one_hot = to_one_hot(self.opt, self.pred, in_channels=self.pred.shape[1])
        self._iou = self.metric_iou(self.pred_one_hot, self.target_one_hot)
        return self._iou

