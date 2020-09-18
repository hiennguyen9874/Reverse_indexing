# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import io
import os
import cv2
import json
import yaml
import torch
import shutil
import subprocess
import collections
import pkg_resources
import collections.abc

import torch
import pickle
import torch.nn as nn

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np

from PIL import Image

from tqdm.auto import tqdm
from pathlib import Path
from collections import OrderedDict
from torch.utils.data.dataloader import DataLoader

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
config = read_config('config.yml', False)


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
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = imread(img_path)
        result = self.transform(image=img)
        return result['image'], label, img_path

    def __len__(self):
        return len(self.data)

dataset = ImageDataset(data=datasource.get_data('test'), transform=image_processing)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)


# %%
all_data = []

with torch.no_grad():
    for batch_idx, (image, _labels, img_path) in tqdm(enumerate(dataloader)):
        image = image.to(device)

        out = model(image)
        out = torch.sigmoid(out)

        out[out>0.75]=2
        out[out<0.25]=0
        out[(out >= 0.25) & (out <= 0.75)] = 1
        
        out = out.type(torch.IntTensor).tolist()

        all_data.extend(list(zip(img_path, out)))
        # for idx in range(len(img_path)):
        #     all_data.append((img_path[idx], out[idx]))

        if batch_idx == 10:
            break

# %%
all_data


# %%
database.remove_all()
time_insert = database.insert(data=all_data, attribute_label=attribute_name, tag='pre_insert')
print(f'time insert data: {time_insert}')

# %%
query_dict = {'personalMale': 0}

# %%
def get_query_key(query_dict, tag=None):
    r""" Return query string from dict.
    """
    query_key = ''
    for attribute in attribute_name:
        query_key += attribute + '-'
        if attribute in query_dict.keys():
            query_key += str(query_dict[attribute])
        else:
            query_key += '?'
        query_key += '_'
    if tag != None:
        query_key += 'tag-'+ tag + '_'
    query_key = query_key + '*'
    return query_key

query_str = get_query_key(query_dict)
query_str


# %%
all_path = database.query_all(query_str)
all_path


# %%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

list_path = all_path[:10]

img = np.concatenate([Image.open(x).resize((64, 128)) for x in list_path], axis=1)
plt.figure(figsize=(40, 20*5))
plt.imshow(img)
plt.axis('off')
plt.show()


# %%



# %%



