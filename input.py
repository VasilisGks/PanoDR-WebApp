from helpers import maskHandler
from panorama import *
import threading
import numpy as np
from numpy.lib import mask_indices
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
from PIL import Image, ImageChops
import cv2
from data import *
import json
import io
from helpers import *
import logging
from lxml import html
from streamlit_drawable_canvas import st_canvas
from streamlit import components
import base64
import ast
import argparse
import ast
from helpers import initCanvasParams
from data import *
from models.PanoDR.PanoDR_module import *
from models.PanoDR.PanoPaint_module import *
import poissonimageeditting as poisson
logger = logging.getLogger(__name__)

def ApplyCanvas(stroke_width, stroke_color, bg_color, bg_image, drawing_mode, realtime_update, background_image):

    canvas_result = st_canvas(
        fill_color="rgb(0, 0, 0)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=256,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    st_canvas(initial_drawing=canvas_result.json_data)
    return bg_image, canvas_result

def DR_service(
    args: argparse.Namespace, 
    device: str,
    model_option: str
    ) -> None:
        
    stroke_width, stroke_color, bg_color, bg_image, drawing_mode, realtime_update, background_image = initCanvasParams()
    if st.sidebar.checkbox('Use image blending', value=True):
        args.use_blending = True
    if st.sidebar.checkbox('Upload from local folder'):
        bg_image, canvas_result = ApplyCanvas(stroke_width, stroke_color, bg_color, bg_image, drawing_mode, realtime_update, background_image)
    else:
        content_name = st.sidebar.selectbox("Choose the content images:", input_images)
        content_file = input_images_dict[content_name]
        bg_image, canvas_result = ApplyCanvas(stroke_width, stroke_color, bg_color, content_file, drawing_mode, realtime_update, background_image)

    #If panorama is given as input, continue
    if bg_image is not None and st.button('Run '):
        use_resize = True
        background_image = readImg(args, bg_image, use_resize, height=args.height, width=args.width)
        
        with st.spinner('Running DR service..'):
            if args.mask_method == 'FreeForm':
                mask = canvas_result.image_data
                mask = maskHandler(mask, height=args.height, width=args.width)

            background_image_t, input_mask = preProcess(background_image, mask)
            background_image_t = background_image_t.to(device)
            input_mask = input_mask.to(device)

            PanoDRmodel = load_network(args, device, args.model_type)
            if model_option == 'PanoDR':
                raw, pred, masked_input, layout = PanoDRmodel.inference_file_dr(background_image_t, input_mask)
            elif model_option == 'PanoPaint': 
                raw, pred, masked_input, layout = PanoDRmodel.inference_file_inpaint(background_image_t, input_mask)
            pred  = torch.from_numpy(pred)
            pred = pred.unsqueeze(0).permute(0,3,1,2)
            pred = (pred * input_mask) + (background_image_t * (1.0 - input_mask)) 
            pred = pred.squeeze(0).permute(1,2,0).cpu().numpy()
            
            if args.use_blending:
                blended, overlapped = poisson.poisson_blend(pred, input_mask.squeeze(0).squeeze(0).cpu().numpy(), pred, 'mix', output_dir=None)
                
            st.image(masked_input, clamp=True, caption='Masked Input Image')
            st.image(pred, clamp=True, caption='Predicted Diminished Panorama')
            if args.use_blending:
                st.image(blended, clamp=True, caption='Blended Panorama')
            st.image(layout, clamp=True, caption='Predicted Input Dense layout')

@st.cache(allow_output_mutation=True, ttl=3600, max_entries=1)
def load_network(
    args: argparse.Namespace, 
    device: str,
    model_option = None
    ) -> None:
    
    if model_option == 'PanoDR':
        PanoDRmodel = PanoDR(opt=args, device=device)
        try:
            checkpoint_segm = torch.hub.load_state_dict_from_url(args.segmentation_model_chkpnt_dr, map_location=device)
        except:
            checkpoint_segm = torch.load(args.segmentation_model_chkpnt_dr, map_location=torch.device(device))
            
    elif model_option =='PanoPaint':
        PanoDRmodel = InpaintingModel(opt=args, device=device)
        checkpoint_segm = torch.load(args.segmentation_model_chkpnt, map_location=torch.device(device))
    if args.use_LS == False:
        PanoDRmodel.netG.structure_model.load_state_dict(checkpoint_segm)

    PanoDRmodel = PanoDRmodel.to(device)
        
    return PanoDRmodel
    
    