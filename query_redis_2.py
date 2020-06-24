import time
import redis
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from data import Market1501_Attribute, PA_100K

query = {'Shorts': 1, 'Female': 1, 'Age18-60': 1, 'Backpack': 1}

datasets = 'PA-100K'

if datasets == 'Market-1501':
    attribute_label = Market1501_Attribute(
        root_dir='/home/hien/Documents/datasets', download=False, extract=False, re_label_on_train=False).get_attribute('train')[1]
elif datasets == 'PA-100K':
    attribute_label = PA_100K(root_dir='/home/hien/Documents/datasets', download=False, extract=False).attr_name
    
#   ['Female', 'AgeOver60', 'Age18-60', 'AgeLess18', 'Front', 'Side', 'Back', 'Hat', 'Glasses',
#   'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront', 'ShortSleeve', 'LongSleeve',
#   'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice', 'LowerStripe', 'LowerPattern',
#   'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress', 'boots']

query_key = ''
for attribute in attribute_label:
    query_key += attribute + '-'
    if attribute in query.keys():
        query_key += str(query[attribute])
    else:
        query_key += '*'
    query_key += '_'
query_key += '*'

r = redis.StrictRedis(host='localhost', port=6379, db=1)

start_time = time.time()
list_path = []
all_keys = r.keys(query_key)
for key in all_keys:
    list_path.append(r.hgetall(key)['path'.encode('utf-8')].decode('utf-8'))
print("time excute: %s seconds" % (time.time() - start_time))

print(f'num of result: {len(list_path)}')

num_img = 5
random.shuffle(list_path)
start_int = random.randint(0, len(list_path)-1)
new_list_path = list_path[start_int:start_int+num_img]

list_img = []
for path in new_list_path:
    img = Image.open(path)
    img = img.resize((64, 128))
    list_img.append(img)

if len(list_img) != 0:
    img = np.concatenate(list_img, axis=1)
    plt.figure(figsize=(80, 40*num_img))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
print('Done!')

# time excute: 0.43029260635375977 seconds
