import time
from pymongo import MongoClient
from data import Market1501_Attribute

datasource = Market1501_Attribute(root_dir='/home/hien/Documents/datasets', download=True, extract=True, re_label_on_train=False)

client = MongoClient('mongodb://admin:password@localhost')

client.drop_database('Market1501')
db = client['Market1501']

collection = db.train

start_time = time.time()
attribute_label = datasource.get_attribute('train')[1]
list_save_dict = list()
for index in range(len(datasource.get_data('train'))):
    # collection.save({'_id': datasource.get_data('train')[index][0], 'pid': datasource.get_data('train')[index][1], 'camid': datasource.get_data('train')[index][2]})
    save_dict = {
        '_id': datasource.get_data('train')[index][0],
        'person_id': datasource.get_data('train')[index][1],
        'camera_id': datasource.get_data('train')[index][2]}
    
    for index_attribute in range(len(attribute_label)):
        save_dict.update({attribute_label[index_attribute]: int(datasource.get_attribute('train')[0][datasource.get_data('train')[index][1]][index_attribute])})
        # print(attribute_label[index_attribute])
        # print(datasource.get_attribute('train')[0][datasource.get_data('train')[index][1]][index_attribute])
    # collection.insert_one(save_dict)
    list_save_dict.append(save_dict)
collection.insert_many(list_save_dict)
print("time excute: %s seconds" % (time.time() - start_time))