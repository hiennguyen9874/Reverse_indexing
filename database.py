import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))

import time
import redis

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from datetime import datetime
from collections import deque

from data import PA_100K, Peta

__all__ = ['Database_reid']

class Database_reid(object):
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        self.r = redis.Redis(host=host, port=port, db=db, password=password)
        self.pipe = self.r.pipeline()
        self.attribute_label = None
    
    def is_redis_available(self):
        return self.r.ping()
    
    def insert(self, data, attribute_label, tag=None):
        r""" Insert data into database
        Args:
            data (list of tuple(path, list of label)): [(path1, [1, 0, 0, 1]), ()]
        """
        for index in range(len(data)):
            for idx_attribute in range(len(attribute_label)):
                assert isinstance(data[index][1][idx_attribute], int) or np.issubdtype(data[index][1][idx_attribute], int), 'element in data[:, 1] must be int, not {}'.format(type(data[index][1][idx_attribute]))

        self.attribute_label = attribute_label
        start_time = time.time()
        start_index = int(datetime.now().strftime(r'%m%d%H%M%S'))
        for index in range(len(data)):
            save_dict = {
                str(start_index+index): data[index][0]
            }
            key = ''
            for index_attribute in range(len(attribute_label)):
                key += attribute_label[index_attribute] + '-' + str(data[index][1][index_attribute]) + '_'
            if tag != None:
                key += 'tag-'+ tag + '_'
            self.pipe.hmset(key, save_dict)
        self.pipe.execute()
        return time.time() - start_time
    
    def insert_float(self, data, attribute_label, tag=None):
        r""" Insert data into database
        Args:
            data (list of tuple(path, list of label)): [(path1, [1, 0, 0, 1]), ()]
        """
        self.attribute_label = attribute_label
        start_time = time.time()
        start_index = int(datetime.now().strftime(r'%m%d%H%M%S'))
        for index in range(len(data)):
            save_dict = {
                str(start_index+index): data[index][0]
            }
            key = ''
            for index_attribute in range(len(attribute_label)):
                value_at_attribute = '{:.1f}'.format(data[index][1][index_attribute])
                if value_at_attribute == '1.0':
                    value_at_attribute = '0.a'
                key += attribute_label[index_attribute] + '-' + value_at_attribute + '_'
            if tag != None:
                key += 'tag-'+ tag + '_'
            self.pipe.hmset(key, save_dict)
        self.pipe.execute()
        return time.time() - start_time
    
    def query(self, query_str: str):
        r""" Query theo tung dong, khong xac dinh duoc so anh tra ve, khong cache lai tren local
        """
        cursor1 = '0'
        while cursor1 != 0:
            cursor1, data1 = self.r.scan(cursor=cursor1, match=query_str)
            if len(data1) == 0:
                continue
            for item1 in data1:
                list_value = self.r.hgetall(item1)
                list_path = []
                for key, value in list_value.items():
                    list_path.append(value.decode('utf-8'))
                yield list_path
    
    def new_query(
        self, 
        query_str: str, 
        cursor1 = '0', 
        cached_value = deque(), 
        num_images: int=5, 
        head_data=[],
        num_image_per_key=10):
        
        cursor = cursor1
        cached_value = cached_value
        while len(cached_value) >= num_images:
            return [cached_value.pop().decode('utf-8') for _ in range(num_images)], cursor, cached_value, head_data
        
        if len(head_data) > 0:
            for index, item in enumerate(head_data):
                self.pipe.hgetall(item)
                if index == len(head_data) - 1 or ((index+1)*num_image_per_key) % num_images == 0:
                    cached_value.extendleft([y for x in self.pipe.execute() for y in x.values()])
                while len(cached_value) >= num_images:
                    return [cached_value.pop().decode('utf-8') for _ in range(num_images)], cursor, cached_value, head_data[index+1:]

        index = 0
        while cursor != 0:
            cursor, data = self.r.scan(cursor=cursor, match=query_str)
            for index1, item in enumerate(data):
                self.pipe.hgetall(item)
                index += 1
                if index1 == len(data) - 1 or ((index+1)*num_image_per_key) % num_images == 0:
                    cached_value.extendleft([y for x in self.pipe.execute() for y in x.values()])
                while len(cached_value) >= num_images:
                    return [cached_value.pop().decode('utf-8') for _ in range(num_images)], cursor, cached_value, data[index1+1:]
        
        cached_value.extendleft([y for x in self.pipe.execute() for y in x.values()])
        if len(cached_value) > 0:
            return [cached_value.pop().decode('utf-8') for _ in range(len(cached_value))], cursor, cached_value, []

    def query_fixed_count(self, query_str: str, num_images: int):
        r""" Query theo tung key, khi nao du so luong thi tra ve, cache lai phan du tren ram
        Args:
            query_str (str): query string from users
            num_images (int): num of images in Paging
        Returns:
            list of path to image
        """
        cached_value = deque()
        for index, key in enumerate(self.r.scan_iter(match=query_str)):
            self.pipe.hgetall(key)
            if (index+1) % num_images == 0:
                cached_value.extendleft([y for x in self.pipe.execute() for y in x.values()])
            while len(cached_value) >= num_images:
                yield [cached_value.pop().decode('utf-8') for _ in range(num_images)]
        cached_value.extendleft([y for x in self.pipe.execute() for y in x.values()])
        if len(cached_value) > 0:
            yield [cached_value.pop().decode('utf-8') for _ in range(len(cached_value))]
        
    def query_with_num(self, query_str: str, num_images):
        r""" query tat ca key match, cache lai toan bo key match tren ram, tra ve so luong theo ham python generator
        """
        all_keys = self.r.keys(query_str)
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
        r""" Query all key matched with query_str, return all path
        """
        all_keys = self.r.keys(query_str)
        for key in all_keys:
            self.pipe.hgetall(key)
        return [y.decode('utf-8') for x in self.pipe.execute() for y in x.values()]
    
    def query_all_with_num(self, query_str: str, num_images):
        r""" Query toan bo anh match voi query string, luu lai toan bo path do tren ram, tra ve du so luong theo ham python generator
        """
        all_path = self.query_all(query_str)
        index = 0
        while index < len(all_path):
            list_path = all_path[index:index+num_images]
            index += num_images
            yield list_path

    def get_query_key(self, query_dict, tag=None, with_one=False):
        r""" Return query string from dict.
        """
        assert all(map(lambda x: type(x) == bool, query_dict.values()))
        def convert_value(value, with_one):
            if value:
                if with_one:
                    return "[21]"
                else:
                    return "2"
            else:
                if with_one:
                    return "[01]"
                else:
                    return "0"
        query_key = ''
        for attribute in self.attribute_label:
            query_key += attribute + '-'
            if attribute in query_dict.keys():
                query_key += convert_value(query_dict[attribute], with_one)
            else:
                query_key += '?'
            query_key += '_'
        if tag != None:
            query_key += 'tag-'+ tag + '_'
        query_key = query_key + '*'
        return query_key
    
    def set_attribute_label(self, attribute_label):
        self.attribute_label = attribute_label
    
    def remove_all(self):
        r""" Remove all in current database
        """
        self.r.flushdb()

if __name__ == "__main__":
    print('Connecting...')
    database = Database_reid(host='52.230.123.142', port=6379, db=10, password='abcxyz123')
    print('Connected!')
    
    datasource = PA_100K('/datasets', False, False, True)
    
    # set attribute into database
    attribute_label = datasource.get_attribute()
    print("num attribute: %d" % (len(attribute_label)))
    database.set_attribute_label(attribute_label)

    # # remove all in database
    # database.remove_all()
    # print('remove all in database!')

    # # insert
    # all_data = datasource.get_data('train') + datasource.get_data('val') + datasource.get_data('test')
    # new_all_data = []
    # for img_path, labels in all_data:
    #     new_labels = np.rint(labels).astype(np.int)
    #     new_labels[new_labels == 1] = 2
    #     new_all_data.append((img_path, new_labels))
    # time_insert = database.insert(data=new_all_data, attribute_label=attribute_label, tag='pre_insert')
    # print(f'time insert data: {time_insert}')
    
    num_img = 10
    query_dict = {'AgeLess18': True, 'Female': True}
    query_str = database.get_query_key(query_dict, with_one=True)
    
    time_start = time.time()
    all_path = database.query_all(query_str)
    print('time query all: ', time.time()-time_start)
    print('len: ', len(all_path))

    # for list_path in database.query_fixed_count(query_str, num_img):
    #     img = np.concatenate([Image.open(x).resize((64, 128)) for x in list_path], axis=1)
    #     plt.figure(figsize=(40, 20*num_img))
    #     plt.imshow(img)
    #     plt.axis('off')
    #     plt.show()

    # cursor = '0'
    # cached_value = deque()
    # head_data = []
    
    # all_path = []

    # all_time = 0.0
    # all_count = 0
    # time_start = time.time()

    # while cursor != 0 or len(cached_value) > 0:
    #     list_path, cursor, cached_value, head_data = database.new_query(
    #         query_str,
    #         cursor1=cursor, 
    #         num_images=num_img, 
    #         head_data=head_data
    #     )
    #     all_path.extend(list_path)
    #     all_time += time.time() - time_start
    #     time_start = time.time()
    #     all_count += 1
        
    #     # img = np.concatenate([Image.open(x).resize((64, 128)) for x in list_path], axis=1)
    #     # plt.figure(figsize=(40, 20*num_img))
    #     # plt.imshow(img)
    #     # plt.axis('off')
    #     # plt.show()
    # print(all_time/all_count)

