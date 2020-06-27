
import numpy as np
import scipy.io
import glob
import re
import zipfile
import tarfile
import requests
import os
import sys
sys.path.append('.')
from tqdm import tqdm, tnrange
from collections import defaultdict
from utils import download_file_from_google_drive, download_with_url
class PA_100K(object):
    dataset_dir = 'pa_100k'
    dataset_id = '13UjvKJQlkNXAmvsPG6h5dwOlhJQA_TcT'
    file_name = 'PA-100K.zip'
    google_drive_api = 'AIzaSyAVfS-7Dy34a3WjWgR509o-u_3Of59zizo'
    
    def __init__(self, root_dir='datasets', download=True, extract=True):
        self.root_dir = root_dir
        if download:
            print("Downloading!")
            self._download()
            print("Downloaded!")
        if extract:
            print("Extracting!")
            self._extract()
            print("Extracted!")

        data_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed')

        self.pid_container = dict()
        self.camid_containter = dict()
        self.frames_container = dict()
        pid2label = dict()

        f = scipy.io.loadmat(os.path.join(data_dir, 'annotation.mat'))
        image_name = dict()
        label = dict()
        image_name['train'] = [os.path.join(
            data_dir, 'images', f['train_images_name'][i][0][0]) for i in range(80000)]
        label['train'] = f['train_label']
        image_name['val'] = [os.path.join(
            data_dir, 'images',  f['val_images_name'][i][0][0]) for i in range(10000)]
        label['val'] = f['val_label']
        image_name['test'] = [os.path.join(
            data_dir, 'images', f['test_images_name'][i][0][0]) for i in range(10000)]
        label['test'] = f['test_label']

        self.attr_name = [f['attributes'][i][0][0] for i in range(26)]
        self.train = list(zip(image_name['train'], label['train']))
        self.val = list(zip(image_name['val'], label['val']))
        self.test = list(zip(image_name['test'], label['test']))

    def get_data(self, mode='train'):
        if mode == 'train':
            return self.train, self.attr_name
        elif mode == 'val':
            return self.val, self.attr_name
        elif mode == 'test':
            return self.test, self.attr_name
        else:
            raise ValueError('mode error')

    def _process_dir(self, path, relabel):
        pass

    def _download(self):
        os.makedirs(os.path.join(self.root_dir,
                                 self.dataset_dir, 'raw'), exist_ok=True)
        download_with_url(self.google_drive_api, self.dataset_id, os.path.join(self.root_dir, self.dataset_dir, 'raw'), self.file_name)

    def _extract(self):
        file_path = os.path.join(
            self.root_dir, self.dataset_dir, 'raw', self.file_name)
        extract_dir = os.path.join(
            self.root_dir, self.dataset_dir, 'processed')
        if self._exists(extract_dir):
            return
        try:
            tar = tarfile.open(file_path)
            os.makedirs(extract_dir, exist_ok=True)
            for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                tar.extract(member=member, path=extract_dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(file_path, 'r')
            for member in tqdm(iterable=zip_ref.infolist(), total=len(zip_ref.infolist())):
                zip_ref.extract(member=member, path=extract_dir)
            zip_ref.close()

    def _exists(self, extract_dir):
        if os.path.exists(os.path.join(extract_dir, 'images')) \
                and os.path.exists(os.path.join(extract_dir, 'README_0.txt')) \
                and os.path.exists(os.path.join(extract_dir, 'README_1.txt')) \
                and os.path.exists(os.path.join(extract_dir, 'annotation.mat')):
            return True
        return False

    def get_num_classes(self, datasets: str):
        pass

    def get_num_camera(self, datasets: str):
        pass

    def get_name_dataset(self):
        pass

if __name__ == "__main__":
    datasource = PA_100K(root_dir='/home/hien/Documents/datasets', download=True, extract=True)
    all_data = datasource.get_data('train')[0] + datasource.get_data('val')[0] + datasource.get_data('test')[0]
    pass