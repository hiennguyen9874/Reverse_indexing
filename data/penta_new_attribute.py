import sys
sys.path.append('.')

import os
import requests
import tarfile
import zipfile
import re
import glob
import scipy.io
import numpy as np
from tqdm import tqdm, tnrange  
from collections import defaultdict

from utils import download_with_url

class Penta_New_Attribute(object):
    dataset_dir = 'penta_new_attribute'
    dataset_id = '13UvQ4N-sY67htGnK6qheb027XuMx9Jbr'
    file_name = 'PETA-New.zip'
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
        f = scipy.io.loadmat(os.path.join(data_dir, 'PETA.mat'))

        self.data = list()
        for idx in range(len(f['peta'][0][0][0])):
            self.data.append((os.path.join(data_dir, 'images', '%05d.png'%(f['peta'][0][0][0][idx][0])), f['peta'][0][0][0][idx][4:].tolist()))
        
        self.attribute_label = list()
        for idx in range(len(f['peta'][0][0][1])):
            self.attribute_label.append(f['peta'][0][0][1][idx][0][0])

    def get_data(self, mode='train'):
        return self.data, self.attribute_label

    def get_attribute(self, mode = 'train'):
        self.attribute_label

    def _download(self):
        os.makedirs(os.path.join(self.root_dir, self.dataset_dir, 'raw'), exist_ok=True)
        download_with_url(self.google_drive_api, self.dataset_id, os.path.join(self.root_dir, self.dataset_dir, 'raw'), self.file_name)

    def _extract(self):
        file_path = os.path.join(self.root_dir, self.dataset_dir, 'raw', self.file_name)
        extract_dir = os.path.join(self.root_dir, self.dataset_dir, 'processed')
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
            and os.path.exists(os.path.join(extract_dir, 'README')) \
            and os.path.exists(os.path.join(extract_dir, 'PETA.mat')):
            return True
        return False


if __name__ == "__main__":
    datasource = Penta_New_Attribute(root_dir='/home/hien/Documents/datasets')
    path = datasource.get_data()[0][0]
