import time
import redis
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from data import Market1501_Attribute

r = redis.StrictRedis(host='localhost', port=6379, db=0)

query = {'Shorts': 1, 'Female': 1, 'AgeLess18': 1, 'Backpack': 1}

arr = None

start_time = time.time()
for key, value in query.items():
    if arr is None:
        arr = np.asarray([int(x) for x in r.lrange(str(value) + '_' + key, start=0, end=-1)])
    else:
        arr = np.intersect1d(arr, np.asarray([int(x) for x in r.lrange(str(value) + '_' + key, start=0, end=-1)]))

list_path = []
for x in arr:
    list_path.append(r.hgetall(int(x))['path'.encode('utf-8')].decode('utf-8'))
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

# time excute: 0.7510411739349365 seconds
