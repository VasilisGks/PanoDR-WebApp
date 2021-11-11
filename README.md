# Web application for PanoDR: Spherical Panorama Diminished Reality for Indoor scenes, CVPRW 2021 model 


## Prerequisites
- Windows10 or Linux
- Python 3.7
- NVIDIA GPU + CUDA CuDNN or CPU
- PyTorch 1.7.1 (or higher)

## Installation
- Clone this repo:

```bash
git clone https://github.com/VasilisGks/PanoDR_web_app.git
cd PanoDR_web_app
```

- We recommend setting up a virtual environment (follow the `virtualenv` documentation).
Once your environment is set up and activated, install the requirements via:

```bash
pip install -r requirements.txt
```

## Run
In order to run the Web App simply run:
```bash
streamlit run app.py
```
and open 
```bash
http://localhost:8501/
```
in your browser. You can either select a panorama from the list, or upload one of your choice.

## Web application usage example:
![](https://github.com/VCL3D/PanoDR/blob/gh-pages/assets/web_app.gif) <br />
