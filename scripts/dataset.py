import cv2 as cv
import os
import pandas as pd
import numpy as np
import gc
from typing import Literal, Union
from tqdm import trange

DatasetType = Literal['local', 'csv']
DatasetPartition = Literal['train', 'val', 'test']


class ClassMapper:
    def __init__(self, classes: list[str]):
        self.classes = sorted(classes)
        self.mapper = {}
        self._load_mapper()

    def _load_mapper(self):
        index = 0
        for class_name in self.classes:
            self.mapper[class_name] = index
            index += 1

    def encode(self, class_name: str):
        return self.mapper[class_name]

    def decode(self, index: int):
        return self.classes[index]


class Dataset:
    def __init__(self,
                 dataset_path: str,
                 dataset_mode: DatasetType = "local",
                 class_mapper: ClassMapper = None,
                 shuffle=True,
                 preload=False,
                 partition_name: DatasetPartition = None):
        self.dataset_path = dataset_path
        self.dataset_mode = dataset_mode
        self.preload = preload
        self.metadata: pd.DataFrame = None
        self.index = 0
        self.classes = []
        self.cache = []
        self.class_mapper = class_mapper
        self.shuffle = shuffle
        self.partition_name = partition_name
        self._load_dataset()
        self.classes = sorted(self.metadata['class_name'].unique())
        self._encode_labels()
        self._shuffle_dataset()
        self._preload_dataset()

    def _load_dataset(self):
        print(
            f"[LOG] Importing data from {self.dataset_path}, mode = {self.dataset_mode}")
        # CSV mode reading
        if self.dataset_mode == "csv":
            self.metadata = pd.read_csv(self.dataset_path, sep=";")
            assert "image_name" in self.metadata.columns, "column 'image_name' is required."
            assert "class_name" in self.metadata.columns, "column 'class_name' is required."
            assert "image_path" in self.metadata.columns, "column 'image_path' is required."
            assert "partition_name" in self.metadata.columns, "column 'partition_name' is required."
            if self.partition_name != None:
                print(
                    f"[LOG] Filtered by partition = {self.partition_name}")
                self.metadata = self.metadata[self.metadata["partition_name"]
                                              == self.partition_name]
            return
        classes = os.listdir(self.dataset_path)
        self.metadata = {
            'image_name': [],
            'class_name': [],
            'image_path': [],
        }
        for class_name in classes:
            class_path = os.path.join(self.dataset_path, class_name)
            images = os.listdir(class_path)
            for image_name in images:
                image_path = os.path.join(class_path, image_name)
                self.metadata["class_name"].append(class_name)
                self.metadata["image_name"].append(image_name)
                self.metadata["image_path"].append(image_path)
        self.metadata = pd.DataFrame.from_dict(self.metadata)

    def _encode_labels(self):
        if self.class_mapper == None:
            self.class_mapper = ClassMapper(self.classes)
        if 'label' not in self.metadata.columns:
            self.metadata['label'] = self.metadata['class_name'].map(
                self.encode_label)

    def _shuffle_dataset(self):
        if self.shuffle:
            self.metadata = self.metadata.sample(frac=1)

    def _preload_dataset(self):
        if not self.preload:
            return
        print(f"[LOG] Preloading the dataset")
        for index in trange(len(self.metadata)):
            self.cache.append(self._load_sample(index))

    def load_sample_from_dataframe(self, sample: pd.Series):
        y = self.encode_label(sample['class_name'])
        image_name = sample['image_name']
        image_path = sample['image_path']
        x = cv.imread(image_path)
        x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        return x, y, image_name

    def free(self):
        print("[LOG] Deallocating memory and turning preload off.")
        del self.cache
        self.preload = False
        self.cache = []
        gc.collect()

    def _load_sample(self, index: int):
        sample = self.metadata.iloc[index]
        return self.load_sample_from_dataframe(sample=sample)

    def encode_label(self, class_name: str):
        return self.class_mapper.encode(class_name=class_name)

    def decode_label(self, index: int):
        return self.class_mapper.decode(index=index)

    def __len__(self):
        return len(self.metadata)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        index = self.index
        self.index += 1
        if self.preload:
            return self.cache[index]
        return self._load_sample(index)
