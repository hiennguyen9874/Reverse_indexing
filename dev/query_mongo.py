import time
import pprint
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pymongo import MongoClient

from data import Market1501_Attribute

client = MongoClient('mongodb://admin:password@localhost')
db = client['Market1501']

collection = db.train

# print(f'num of samples: {db.train.find().count()}')
list_img = list()
start_time = time.time()

pipeline = [
    {'$match:': {'bag': 0, 'hat': 1, 'gender': 1, 'clothes': 0}},
    {'$sort': {'person_id': 1}}
]

for x in collection.find({'bag': 0, 'hat': 1, 'gender': 1, 'clothes': 0}):
    img = Image.open(x['_id'])
    list_img.append(img)
print("time excute: %s seconds" % (time.time() - start_time))

random.shuffle(list_img)
print(f'num of result: {len(list_img)}')
if len(list_img) != 0:
    num_img = 5
    start_int = random.randint(0, len(list_img)-1)
    img = np.concatenate(list_img[start_int:start_int + num_img], axis=1)
    plt.figure(figsize=(80, 40*num_img))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
print('Done!')

