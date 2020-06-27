import time
import redis
from datetime import datetime
from data import Market1501_Attribute, PA_100K

r = redis.StrictRedis(host='localhost', port=6379, db=1)
pipe = r.pipeline()
for key in r.scan_iter("*"):
    pipe.delete(key)
pipe.execute()

datasets = 'PA-100K'

if datasets == 'Market-1501':
    datasource = Market1501_Attribute(
        root_dir='/home/hien/Documents/datasets', download=False, extract=False, re_label_on_train=False)

    all_data = datasource.get_data('train') + datasource.get_data('query') + datasource.get_data('gallery')
    all_attribute = datasource.get_attribute('train')[0]
    all_attribute.update(datasource.get_attribute('query')[0])
    attribute_label = datasource.get_attribute('train')[1]

    start_time = time.time()
    start_index = int(datetime.now().strftime(r'%m%d%H%M%S'))
    for index in range(len(all_data)):
        if all_data[index][1] not in all_attribute.keys():
            continue
        save_dict = {
            'path': all_data[index][0],
            'person_id': all_data[index][1],
            'camera_id': all_data[index][2]}
        key = ''
        for index_attribute in range(len(attribute_label)):
            key += attribute_label[index_attribute] + '-' + str(all_attribute[all_data[index][1]][index_attribute]) + '_'
        pipe.hmset(key + str(start_index+index), save_dict)
    pipe.execute()
    print("time excute: %s seconds" % (time.time() - start_time))
    
elif datasets == 'PA-100K':
    datasource = PA_100K(root_dir='/home/hien/Documents/datasets', download=False, extract=False)
    all_data = datasource.get_data('train')[0] + datasource.get_data('val')[0] + datasource.get_data('test')[0]
    attribute_label = datasource.attr_name
    start_time = time.time()
    start_index = int(datetime.now().strftime(r'%m%d%H%M%S'))
    for index in range(len(all_data)):
        save_dict = {
            'path': all_data[index][0]
        }
        key = ''
        for index_attribute in range(len(attribute_label)):
            key += attribute_label[index_attribute] + '-' + str(all_data[index][1][index_attribute]) + '_'
        pipe.hmset(key + str(start_index+index), save_dict)
    pipe.execute()
    print("time excute: %s seconds" % (time.time() - start_time))

# time excute: 6.300401926040649 seconds