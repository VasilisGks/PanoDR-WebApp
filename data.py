import os

input_images_path = 'images'
source_file_names = ['GS__0020.JPG', 'GS__0004.JPG', 'GS__0021.JPG', 'rgb_03498.png', 'rgb0.png']
input_images = ['lab_in_the_wild_1', 'lab_in_the_wild_2', 'lab_in_the_wild_3', 'Structured3D_1', 'Structured3D_2']
input_images_dict = {name: os.path.join(input_images_path, f_name) for name, f_name in zip(input_images, source_file_names)}