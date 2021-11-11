import numpy as np 
import cv2
from PIL import Image
import torch
import io
from io import StringIO
import requests
from requests.api import head
import streamlit as st
import logging
import json
from data import *
import ast
import PIL
from typing import Tuple, List
from panorama import *
import argparse


def initCanvasParams(
    ) -> Tuple[int, str, str, str, str, bool, None]:
    stroke_width = st.sidebar.slider("Stroke width: ", 10, 35, 20)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#000")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#fff")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
    )
    realtime_update = st.sidebar.checkbox("Update in realtime", False)
    background_image = None
    
    return stroke_width, stroke_color, bg_color, bg_image, drawing_mode, realtime_update, background_image

def preProcess(
    img: torch.tensor,
    mask: torch.tensor
)-> Tuple[torch.tensor, torch.tensor]:

    mask_2d = mask[:,:,0]
    input_mask = torch.from_numpy(mask_2d).unsqueeze_(0).unsqueeze_(0) / 255.0
    background_image = np.array(img, dtype=np.float32) / 255.0
    background_image_t = torch.from_numpy(background_image).unsqueeze_(0).permute(0,3,1,2)
    return background_image_t, input_mask

def maskHandler(
    mask: np.ndarray
    ) -> np.ndarray:

    mask = mask[:,:,3].astype(np.float32)
    thresh = 127
    im_bw = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)[1]
    im_bw = np.tile(im_bw[:, :, None], [1, 1, 3])
    mask = im_bw 
    mask = cv2.resize(mask, (512,256), interpolation=cv2.INTER_NEAREST)
    return mask

@st.cache(suppress_st_warning=True)
def getDevice(
    gpu_id: int
    ) -> torch.device:
    device = torch.device("cuda:" + str(gpu_id) if (torch.cuda.is_available() and int(gpu_id) >= 0) else "cpu")  
    return device

@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)
def readImg(
    args: argparse.Namespace, 
    bg_image: PIL.Image.Image,
    use_resize: bool = True
    )-> Tuple[PIL.Image.Image, io.BytesIO]:

    background_image=Image.open(bg_image)
    background_image = background_image.convert("RGB")
    if use_resize:
        background_image = background_image.resize((512, 256), Image.BICUBIC)

    return background_image 
