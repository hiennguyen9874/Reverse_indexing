
# %%
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))

import io
import os
import cv2
import json
import yaml
import shutil
import subprocess
import collections
import pkg_resources
import collections.abc

import torch
import pickle
import torch.nn as nn
# import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

from data import PA_100K, Peta
from database import *
from models import *
from utils import *

# %%
print('Connecting...')
database = Database_reid(host='52.230.123.142', port=6379, db=10, password='abcxyz123')
print('Connected!')

# %%
datasource = Peta('/datasets', True, True, True)
attribute_name = datasource.get_attribute()

# %%
config = read_config('base_extraction.yml', False)
config
# %%
use_gpu = config['n_gpu'] > 0 and torch.cuda.is_available()
device = torch.device('cuda:0' if use_gpu else 'cpu')
map_location = "cuda:0" if use_gpu else torch.device('cpu')

model, _ = build_model(config, num_classes=len(attribute_name))
checkpoint = torch.load(config['resume'], map_location=map_location)

model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to(device)

image_processing = A.Compose([
    A.Resize(config['data']['size'][0], config['data']['size'][1]),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2()
])

# %%


def imread(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def cv2_to_bytes(origin_image):
    pil_image = Image.fromarray(origin_image)
    im_file = io.BytesIO()
    pil_image.save(im_file, format="JPEG")
    image_bytes = im_file.getvalue()
    return image_bytes


# %%
from tqdm.auto import tqdm
list_image = []
for idx, (img_path, labels) in tqdm(enumerate(datasource.get_data('train'))):
    image = imread(img_path)
    image_binary = cv2_to_bytes(image)
    list_image.append((image_binary, img_path))

# %%
import grequests
url = 'http://52.230.2.212:10040/predictions/eager_model'

list_request = [grequests.post(url, data=image_binary) for image_binary,_ in  list_image]

def exception_handler(request, exception):
    print("Request failed")

all_data = []
step_size = 512
for idx in tqdm(range(len(list_request)//step_size+1)):
    result = grequests.map(tuple(list_request[idx*step_size:(idx+1)*step_size]), size=128, exception_handler=exception_handler)
    for idx in range(len(result)):
        out = list(result[idx].json())

        out = np.array(out)
        out[out > 0.7] = 2
        out[out < 0.3] = 0
        out[(out >= 0.3) & (out <= 0.7)] = 1
        out = out.astype(np.ubyte)

        all_data.append((list_image[idx][1], out.tolist()))

# %%

# %%
database.remove_all()
time_insert = database.insert(data=all_data, attribute_label=attribute_name, tag='pre_insert')
print(f'time insert data: {time_insert}')

# %%
query_dict = {'personalMale': 2, 'accessoryHat': 2, 'personalLess30': 2}


# %%
query_str = '*personalMale-[0]_*'
all_path = database.query_all(query_str)
print(len(all_path))

# %%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

list_path = all_path[:5]

img = np.concatenate([Image.open(x).resize((64, 128)) for x in list_path], axis=1)
plt.figure(figsize=(40, 20*5))
plt.imshow(img)
plt.axis('off')
plt.show()


# %%
