import time
import redis

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from datetime import datetime
from collections import deque


from data import PA_100K, Peta

class Database_reid(object):
    def __init__(self, host='localhost', port=6379, db=0):
        self.r = redis.StrictRedis(host=host, port=port, db=db)
        self.pipe = self.r.pipeline()
        self.attribute_label = None
    
    def insert(self, data, attribute_label):
        self.attribute_label = attribute_label
        start_time = time.time()
        start_index = int(datetime.now().strftime(r'%m%d%H%M%S'))
        for index in range(len(data)):
            save_dict = {
                str(start_index+index): data[index][0]
            }
            key = ''
            for index_attribute in range(len(attribute_label)):
                key += attribute_label[index_attribute] + '-' + str(int(data[index][1][index_attribute])) + '_'
            self.pipe.hmset(key[:-1], save_dict)
        self.pipe.execute()
        return time.time() - start_time
    
    def query(self, query_str: str):
        query_key = self._get_query_key(query_str)
        cursor1 = '0'
        while cursor1 != 0:
            cursor1, data1 = self.r.scan(cursor=cursor1, match=query_key)
            if len(data1) == 0:
                continue
            for item1 in data1:
                list_value = self.r.hgetall(item1)
                list_path = []
                for key, value in list_value.items():
                    list_path.append(value.decode('utf-8'))
                yield list_path
    
    def query_fixed_count(self, query_str: str, num_images: int):
        '''
        Args:
            query_str (str): query string from users
            num_images (int): num of images in Paging
        Returns:
            list of path to image
        '''
        query_key = self._get_query_key(query_str)
        cached_value = deque()
        for index, key in enumerate(self.r.scan_iter(match=query_key)):
            self.pipe.hgetall(key)
            if (index+1) % num_images == 0:
                cached_value.extendleft([y for x in self.pipe.execute() for y in x.values()])
            while len(cached_value) >= num_images:
                yield [cached_value.pop().decode('utf-8') for _ in range(num_images)]
        cached_value.extendleft([y for x in self.pipe.execute() for y in x.values()])
        if len(cached_value) > 0:
            yield [cached_value.pop().decode('utf-8') for _ in range(len(cached_value))]
        
    def query_with_num(self, query_str: str, num_images):
        query_key = self._get_query_key(query_str)
        all_keys = self.r.keys(query_key)
        cached_value = deque()
        for index, key in enumerate(all_keys):
            self.pipe.hgetall(key)
            if (index+1) % num_images == 0:
                cached_value.extendleft([y for x in self.pipe.execute() for y in x.values()])
            while len(cached_value) >= num_images:
                yield [cached_value.pop().decode('utf-8') for _ in range(num_images)]
        cached_value.extendleft([y for x in self.pipe.execute() for y in x.values()])
        if len(cached_value) > 0:
            yield [cached_value.pop().decode('utf-8') for _ in range(len(cached_value))]

    def query_all(self, query_str:str):
        query_key = self._get_query_key(query_str)
        all_keys = self.r.keys(query_key)
        for key in all_keys:
            self.pipe.hgetall(key)
        return [y.decode('utf-8') for x in self.pipe.execute() for y in x.values()]
    
    def query_all_with_num(self, query_str: str, num_images, func):
        all_path = self.query_all(query_str)
        index = 0
        while index < len(all_path):
            list_image = [func(x) for x in all_path[index:index+num_images]]
            index += num_images
            yield list_image

    def _get_query_key(self, query_str):
        query_key = ''
        for attribute in self.attribute_label:
            query_key += attribute + '-'
            if attribute in query_str.keys():
                query_key += str(query_str[attribute])
            else:
                query_key += '*'
            query_key += '_'
        query_key = query_key[:-1] + '*'
        return query_key
    
    def set_attribute_label(self, attribute_label):
        self.attribute_label = attribute_label
    
    def remove_all(self):
        for key in self.r.scan_iter("*", count=10000000):
            self.pipe.delete(key)
        self.pipe.execute()


if __name__ == "__main__":
    print('Connecting...')
    # database = Database_reid(host='168.63.252.148', port=6379, db=0)
    database = Database_reid(host='168.63.252.148', port=6379, db=1)
    print('Connected!')
    
    # datasource = PA_100K('/datasets', True, True, True)
    datasource = Peta('/datasets', True, True, True)
    
    attribute_label = datasource.get_attribute()
    print("num attribute: %d" % (len(attribute_label)))
    database.set_attribute_label(attribute_label)
    
    # remove all in database
    database.r.flushall()
    database.r.flushdb()
    
    # insert
    all_data = datasource.get_data('train') + datasource.get_data('val') + datasource.get_data('test')
    print(f'time insert data: {database.insert(data=all_data, attribute_label=attribute_label)}')
    
    num_img = 10
    query_str = {'hairLong': 1, 'personalMale': 1}

    for list_path in database.query_fixed_count(query_str, num_img):
        img = np.concatenate([Image.open(x).resize((64, 128)) for x in list_path], axis=1)
        plt.figure(figsize=(40, 20*num_img))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

