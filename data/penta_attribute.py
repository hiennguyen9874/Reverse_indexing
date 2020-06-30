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

class Penta_Attribute(object):
    dataset_dir = 'penta_attribute'
    dataset_id = '13OGfT_mvX5VjXW6WdwTLoM1tqEh_N58H'
    file_name = 'PETA.zip'
    google_drive_api = 'AIzaSyAVfS-7Dy34a3WjWgR509o-u_3Of59zizo'
    list_folder = ['3DPeS', 'CAVIAR4REID', 'CUHK', 'GRID', 'i-LID', 'MIT', 'PRID', 'SARC3D', 'TownCentre', 'VIPeR']

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

        self._process_dir(os.path.join(data_dir, '3DPeS', 'archive'), relabel=False)

    def get_data(self, mode='train'):
        pass

    def get_attribute(self, mode = 'train'):
        pass

    def _process_dir(self, path, relabel):
        pass

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
        if os.path.exists(os.path.join(extract_dir, 'ReadMe.txt')):
            for x in self.list_folder:
                if not os.path.exists(os.path.join(extract_dir, x)):
                    return False
            return True
        return False

    def get_num_classes(self, datasets: str):
        pass

    def get_num_camera(self, datasets: str):
        pass

    def get_name_dataset(self):
        pass

if __name__ == "__main__":
    datasource = Penta_Attribute(root_dir='/home/hien/Documents/datasets')