import streamlit as st

from data import *
from input import DR_service
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from helpers import getDevice
import parser
import argparse
from streamlit.server.server import StaticFileHandler
import torch
from io import StringIO 

@classmethod
def _get_cached_version(cls, abs_path: str):
    with cls._lock:
        return cls.get_content_version(abs_path)

StaticFileHandler._get_cached_version = _get_cached_version


def parseArguments():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--mask_method', type=str, default='FreeForm') 
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=256)
    #PanoDR params
    parser.add_argument('--eval_path', type=str, default='output/')
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--segmentation_model_chkpnt', type = str, default = 'https://github.com/VasilisGks/PanoDR_web_app/releases/download/v.0.1.0/Unet_epoch23.zip', help = 'Save checkpoints here')
    parser.add_argument('--eval_chkpnt_folder', type=str, default='https://github.com/VasilisGks/PanoDR_web_app/releases/download/v.0.1.0/57_net_G.zip')
    parser.add_argument('--phase', type = str, default = 'test', help = 'load model name')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'Adam: weight decay')
    parser.add_argument('--lr_gated', type = float, default = 0.0001, help = 'Adam: weight decay')
    parser.add_argument('--in_layout_channels', type = int, default = 3, help = '')
    parser.add_argument('--in_SEAN_channels', type = int, default = 3, help = '')
    parser.add_argument('--style_code_dim', type = int, default = 512, help = '')
    parser.add_argument('--in_channels', type = int, default = 4, help = '')
    parser.add_argument('--in_spade_channels', type = int, default = 3, help = '')
    parser.add_argument('--in_d_channels', type = int, default = 4, help = '')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output 2D Coords')
    parser.add_argument('--latent_channels', type = int, default = 64, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'spherical', choices=["replicate", "reflection", "spherical", "zero"], help = 'the padding type') 
    parser.add_argument('--activation', type = str, default = 'relu', help = 'the activation type')
    parser.add_argument('--activation_decoder', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: beta 1')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: beta 2')
    parser.add_argument('--structure_model', type=str, default="unet", choices=["unet"])
    parser.add_argument('--pretrain_network', type=int, default=0, help = 'Model is pretrained')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--type_sp', type=str, default='SEAN')
    parser.add_argument('--use_argmax', type=bool, default=True) 
    parser.add_argument('--use_sean', type=bool, default=True) 
    arguments = parser.parse_args()

    return arguments

def main(args):
    st.title("PanoDR Web App")
    st.sidebar.title('Navigation')
    st.sidebar.header('Options')

    device = getDevice(args.gpu_id)
    DR_service(args, device)

if __name__ == '__main__':
    args = parseArguments()
    main(args)