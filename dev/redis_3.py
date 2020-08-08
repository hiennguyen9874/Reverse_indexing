import time
import redis
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from datetime import datetime
from collections import deque
from data import PA_100K

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
                key += attribute_label[index_attribute] + '-' + str(data[index][1][index_attribute]) + '_'
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
    database = Database_reid(host='168.63.252.148', port=6379, db=0)
    print('Connected!')
    
    datasource = PA_100K(root_dir='/datasets', download=True, extract=True)
    
    # attribute_label = datasource.attr_name
    # print("num attribute: %d" % (len(attribute_label)))
    # database.set_attribute_label(attribute_label)
    
    # query_str = {'Female': 0, 'Age18-60': 1, 'Backpack': 1, 'Shorts': 0}

    # list_attribute_rand = np.array(datasource.get_list_attribute_random())
    
    ''' Test time insert, query
    '''
    # list_num_sampler = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000, 500000000, 1000000000]
    # list_num_sampler = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    # # list_time_insert = []
    # list_time_query = []
    # database.r.flushall()
    # database.r.flushdb()
    # num_images = 50
    # for num_sampler in list_num_sampler:
    #     print(f'num_sampler: {num_sampler}')
    #     all_data = list(zip(['path/to/images.jpg']*num_sampler, np.random.randint(0, 2, size=(num_sampler, len(attribute_label))).tolist()))

    #     print(f'time insert data: {database.insert(data=all_data, attribute_label=attribute_label)}')
    #     # list_time_insert.append(database.insert(data=all_data, attribute_label=attribute_label))
    #     database.set_attribute_label(attribute_label)

    #     num_query = 0
    #     start_time = time.time()
    #     database.query_fixed_count(query_str, num_images=num_images)
    #     list_time_query.append(time.time()-start_time)

    #     # print(f'time query data: {time.time() - start_time}')
    #     # list_time_query.append(time.time() - start_time)
    #     database.r.flushall()
    #     database.r.flushdb()
    # # plt.plot(list_num_sampler, list_time_insert, "-b", label='time insert')
    # plt.plot(list_num_sampler, list_time_query, "-r", label='time query')
    # plt.legend(loc="upper left")
    # plt.xlabel('num sampler')
    # plt.ylabel('time (s)')
    # plt.show()

    ''' Test time insert when database available
    '''
    # database.r.flushall()
    # database.r.flushdb()
    # num_sampler = 50000
    # list_time_insert = []
    # list_time_first_query = []
    # num_images = 50
    # num_iter = 100
    # for i in range(num_iter):
    #     print(f'i: {i}')
    #     all_data = list(zip(['path/to/images.jpg']*num_sampler, list_attribute_rand[np.random.choice(list_attribute_rand.shape[0], num_sampler, replace=True), :].tolist()))
    #     # print(f'time insert data: {database.insert(data=all_data, attribute_label=attribute_label)}')
    #     list_time_insert.append(database.insert(data=all_data, attribute_label=attribute_label))
        
    #     database.set_attribute_label(attribute_label)
    #     start_time = time.time()
    #     all_time = 0
    #     num_query = 0
    #     for index, all_keys in enumerate(database.query_fixed_count(query_str, num_images=num_images)):
    #         all_time += time.time()-start_time
    #         start_time = time.time()
    #         num_query += 1
    #     if num_query == 0:
    #         num_query = 1
    #     list_time_first_query.append(all_time/num_query)

        # start_time = time.time()
        # all_path = database.query_all(query_str)
        # list_time_first_query.append((time.time() - start_time)/len(all_path))

    # color = 'tab:red'
    # plt.xlabel('num sampler')
    # plt.ylabel('time insert (s)', color=color)
    # plt.plot([x*num_sampler for x in list(range(num_iter))], list_time_insert, color, label='time insert')
    # plt.tick_params(axis='y', labelcolor=color)
    # plt.tight_layout()
    # plt.show()
    
    # color = 'tab:blue'
    # plt.ylabel('time query (s)', color=color)
    # plt.plot([x*num_sampler for x in list(range(1, num_iter+1))], list_time_first_query, color, label='time query')
    # plt.tick_params(axis='y', labelcolor=color)
    # plt.tight_layout()
    # plt.show()

    # fig1, ax1 = plt.subplots()
    # ax1.plot([x*5 for x in list(range(num_iter))], list_time_insert, '-r', label='time insert')
    # ax1.set_xlabel('num sampler available (x10^4)')
    # ax1.set_xlim(0, (num_iter-1)*5)
    # ax1.set_xticks(np.arange(0, 5 * num_iter, 5))
    # ax1.set_ylabel('time (s)')
    # fig1.tight_layout()

    # fig2, ax2 = plt.subplots()
    # ax2.plot([x*5 for x in list(range(1, num_iter+1))], [x*10000 for x in list_time_first_query] , '-b', label='time query')
    # ax2.set_xlabel('num sampler in database (x10^4)')
    # ax2.set_xlim(5, num_iter*5)
    # ax2.set_xticks(np.arange(5, 5 * (num_iter+1), 5))
    # ax2.set_ylabel('time (x10^-5 s)')
    # fig2.tight_layout()
    # plt.show()

    ''' Test query with num of images
    '''
    # database.r.flushall()
    # database.r.flushdb()
    # all_data = datasource.get_data('train')[0] + datasource.get_data('val')[0] + datasource.get_data('test')[0]
    # print(f'time insert data: {database.insert(data=all_data, attribute_label=attribute_label)}')

    # num_sampler = 1000000
    # database.r.flushall()
    # database.r.flushdb()
    # for _ in range(2):
    #     all_data = list(zip(['path/to/images.jpg']*num_sampler, np.random.randint(0, 2, size=(num_sampler, len(attribute_label))).tolist()))
    #     print(f'insert {num_sampler} samplers')
    #     print(f'time insert data: {database.insert(data=all_data, attribute_label=attribute_label)}')
    # num_sampler *= 2
    
    # print("Option 1")
    # start_time = time.time()
    # num_images = 100
    # num_query = 0
    # all_time = 0.0
    # for index, all_keys in enumerate(database.query_fixed_count(query_str, num_images=num_images)):
    #     if index == 0:
    #         print('index: %d, time query %d images: %f' % (index, num_images, time.time()-start_time))
    #     else:
    #         all_time += time.time()-start_time
    #         num_query += 1
    #     start_time = time.time()
    # print('avg query time: %f' % (all_time/num_query))

    # print('Option 2')
    # start_time = time.time()
    # num_query = 0
    # all_time = 0.0
    # for index, all_keys in enumerate(database.query_with_num(query_str, num_images=num_images)):
    #     if index == 0:
    #         print('index: %d, time query %d images: %f' % (index, num_images, time.time()-start_time))
    #     else:
    #         all_time += time.time()-start_time
    #         num_query += 1
    #     start_time = time.time()
    # print('avg query time: %f' % (all_time/num_query))

    # print("Option 3")
    # start_time = time.time()
    # all_path = database.query_all(query_str)
    # print('time query all images: %f' % (time.time() - start_time))

    ''' Demo query all with num of images, datasets: PA-100K
    '''
    # database.r.flushall()
    # database.r.flushdb()
    # all_data = datasource.get_data('train')[0] + datasource.get_data('val')[0] + datasource.get_data('test')[0]
    # print(f'time insert data: {database.insert(data=all_data, attribute_label=attribute_label)}')
    # num_img = 10

    # for list_path in database.query_fixed_count(query_str, num_img):
    #     img = np.concatenate([Image.open(x).resize((64, 128)) for x in list_path], axis=1)
    #     plt.figure(figsize=(40, 20*num_img))
    #     plt.imshow(img)
    #     plt.axis('off')
    #     plt.show()
