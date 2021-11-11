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
    ) -> None:
        
    stroke_width, stroke_color, bg_color, bg_image, drawing_mode, realtime_update, background_image = initCanvasParams()
    if st.sidebar.checkbox('Upload from local folder'):
        bg_image, canvas_result = ApplyCanvas(stroke_width, stroke_color, bg_color, bg_image, drawing_mode, realtime_update, background_image)
    else:
        content_name = st.sidebar.selectbox("Choose the content images:", input_images)
        content_file = input_images_dict[content_name]
        bg_image, canvas_result = ApplyCanvas(stroke_width, stroke_color, bg_color, content_file, drawing_mode, realtime_update, background_image)

    #If panorama is given as input, continue
    if bg_image is not None and st.button('Run '):
        use_resize = True
        background_image = readImg(args, bg_image, use_resize)
        
        with st.spinner('Running DR service..'):
            if args.mask_method == 'FreeForm':
                mask = canvas_result.image_data
                mask = maskHandler(mask)

            background_image_t, input_mask = preProcess(background_image, mask)
            background_image_t = background_image_t.to(device)
            input_mask = input_mask.to(device)

            PanoDRmodel = load_network(args, device)
            raw_pred, pred, masked_input, layout = PanoDRmodel.inference_file(background_image_t, input_mask, '')

            st.image(masked_input, clamp=True, caption='Masked Input Image')
            st.image(raw_pred, clamp=True, caption='Raw Predicted Diminished Image')
            st.image(pred, clamp=True, caption='Predicted Diminished Panorama')
            st.image(layout, clamp=True, caption='Predicted Input Dense layout')

@st.cache(allow_output_mutation=True, ttl=3600, max_entries=1)
def load_network(
    args: argparse.Namespace, 
    #method: str,
    device: str
    ) -> None:
    PanoDRmodel = PanoDR(opt=args, device=device)

    checkpoint_segm = torch.hub.load_state_dict_from_url(args.segmentation_model_chkpnt, map_location=device)
    PanoDRmodel.netG.structure_model.load_state_dict(checkpoint_segm)

    checkpoint = torch.hub.load_state_dict_from_url(args.eval_chkpnt_folder, map_location=device)
    PanoDRmodel.netG.load_state_dict(checkpoint)

    PanoDRmodel = PanoDRmodel.to(device)

    return PanoDRmodel