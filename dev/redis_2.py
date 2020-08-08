import time
import redis
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from data import Market1501_Attribute, PA_100K

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
                'path': data[index][0]
            }
            key = ''
            for index_attribute in range(len(attribute_label)):
                key += attribute_label[index_attribute] + '-' + str(data[index][1][index_attribute]) + '_'
            self.pipe.hmset(key + str(start_index+index), save_dict)
        self.pipe.execute()
        return time.time() - start_time
    
    def query(self, query_str: str):
        query_key = self._get_query_key(query_str)
        cursor1 = '0'
        while cursor1 != 0:
            cursor1, data1 = self.r.scan(cursor=cursor1, match=query_key)
            if len(data1) == 0:
                continue
            list_path = []
            for item1 in data1:
                list_value = self.r.hgetall(item1)
                for key, value in list_value.items():
                    list_path.append(value.decode('utf-8'))
            yield list_path
    
    def query_all(self, query_str: str):
        query_key = self._get_query_key(query_str)
        all_keys = self.r.keys(query_key)
        for key in all_keys:
            self.pipe.hgetall(key)
        return [y.decode('utf-8') for x in self.pipe.execute() for y in x.values()]

    def _get_query_key(self, query_str):
        query_key = ''
        for attribute in self.attribute_label:
            query_key += attribute + '-'
            if attribute in query_str.keys():
                query_key += str(query_str[attribute])
            else:
                query_key += '*'
            query_key += '_'
        query_key += '*'
        return query_key
    
    def set_attribute_label(self, attribute_label):
        self.attribute_label = attribute_label
    
    def remove_all(self):
        for key in self.r.scan_iter("*", count=1000000):
            self.pipe.delete(key)
        self.pipe.execute()

if __name__ == "__main__":
    datasource = PA_100K(root_dir='/home/hien/Documents/datasets', download=False, extract=False)
    all_data = datasource.get_data('train')[0] + datasource.get_data('val')[0] + datasource.get_data('test')[0]
    attribute_label = datasource.attr_name

    database = Database_reid(host='localhost', port=6379, db=1)
    # database.remove_all()
    # print(f'time insert data: {database.insert(data=all_data, attribute_label=attribute_label)}')
    database.set_attribute_label(attribute_label)

    num_img = 5
    query_str={'Shorts': 1, 'Female': 1, 'Age18-60': 1, 'Backpack': 1}

    all_path = database.query_all(query_str)
    pass

    # for list_path in database.query(query_str=query_str):
    #     list_image = []
    #     for path in list_path:
    #         img = Image.open(path)
    #         img = img.resize((64, 128))
    #         list_image.append(img)
    #     img = np.concatenate(list_image, axis=1)
    #     plt.figure(figsize=(80, 40*5))
    #     plt.imshow(img)
    #     plt.axis('off')
    #     plt.show()
    
# time insert data: 6.266353607177734
