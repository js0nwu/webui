import torch
import os
from PIL import Image
import json
from torchvision import transforms
import pytorch_lightning as pl
import pandas as pd
from collections import Counter
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from random import choices
import random

DEVICE_SCALE = {
    "default": 1,
    "iPad-Mini": 2,
    "iPad-Pro": 2,
    "iPhone-13 Pro": 3,
    "iPhone-SE": 3
}

def makeOneHotVec(idx, num_classes):
    vec = [1 if i == idx else 0 for i in range(num_classes)]
    return vec

def collate_fn_silver(batch):
    res = defaultdict(list)

    for d in batch:
        for k, v in d.items():
            res[k].append(v)

    res['label'] = torch.tensor(res['label'], dtype=torch.long)
    return res

def collate_fn_silver_multi(batch):
    res = defaultdict(list)

    for d in batch:
        for k, v in d.items():
            res[k].append(v)

    res['label'] = torch.stack(res['label'], dim=0)
    return res

def collate_fn(batch):
    res = defaultdict(list)

    for d in batch:
        for k, v in d.items():
            res[k].append(v)

    res['label'] = torch.stack(res['label'])
    return res

def collate_fn_enrico(batch):
    res = defaultdict(list)

    for d in batch:
        for k, v in d.items():
            res[k].append(v)

    res['label'] = torch.tensor(res['label'], dtype=torch.long)
    return res


class EnricoImageDataset(torch.utils.data.Dataset):
    def __init__(self, id_list_path, csv="../../metadata/screenclassification/design_topics.csv", class_map_file="../../metadata/screenclassification/class_map_enrico.json", img_folder = os.environ['SM_CHANNEL_TRAINING'] if 'SM_CHANNEL_TRAINING' in os.environ else "../../downloads/enrico/screenshots", img_size=128, ra_num_ops=-1, ra_magnitude=-1, one_hot_labels=False):
        super(EnricoImageDataset, self).__init__()
        self.csv=pd.read_csv(csv)
        self.img_folder=img_folder
        self.one_hot_labels = one_hot_labels
        img_transforms = [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        if ra_num_ops > 0 and ra_magnitude > 0:
            img_transforms = [transforms.RandAugment(ra_num_ops, ra_magnitude)] + img_transforms

        self.img_transforms = transforms.Compose(img_transforms)

        self.image_names = list(self.csv['screen_id'])
        self.labels = list(self.csv['topic'])
        
        self.class_counter = Counter(self.labels)
        
        with open(id_list_path, "r") as f:
            split_ids = set(json.load(f))
            
        keep_inds = [i for i in range(len(self.image_names)) if str(self.image_names[i]) in split_ids]
        
        self.image_names = [self.image_names[i] for i in keep_inds]
        self.labels = [self.labels[i] for i in keep_inds]

        with open(class_map_file, "r") as f:
            map_dict = json.load(f)

        self.label2Idx = map_dict['label2Idx']
        self.idx2Label = map_dict['idx2Label']

    #The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self,index):
        img_path = os.path.join(self.img_folder, str(self.image_names[index]) + ".jpg")
        image = Image.open(img_path).convert("RGB")
        
        image=self.img_transforms(image)
        targets = self.label2Idx[self.labels[index]]
        if self.one_hot_labels:
            targets=torch.tensor(makeOneHotVec(targets, len(self.idx2Label.keys())), dtype=torch.long)

        return {'image': image,'label':targets}
    

class CombinedImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, ds_list, prob_list):
        super(CombinedImageDataset, self).__init__()
        self.ds_list = ds_list
        self.prob_list = prob_list
    def __iter__(self):
        while True:
            dsi = choices(list(range(len(self.ds_list))), self.prob_list)[0]
            ds = self.ds_list[dsi]
            dse = int(random.random() * len(ds))
            val = ds.__getitem__(dse)
            yield val
    
class SilverMultilabelImageDataset(torch.utils.data.Dataset):
    def __init__(self, id_list_path=None, silver_id_list_path_ignores=None, K=150, P=1, csv="../../metadata/screenclassification/silver_webui-multi_topic.csv", img_folder = os.environ['SM_CHANNEL_TRAINING'] if 'SM_CHANNEL_TRAINING' in os.environ else "../../downloads/ds", img_size=128, one_hot_labels=False, ra_num_ops=-1, ra_magnitude=-1):
        super(SilverMultilabelImageDataset, self).__init__()
        with open(csv, "r") as file:
             first_line = file.readline()
        num_classes = len(first_line.split(",")) - 1
        self.num_classes = num_classes
        self.one_hot_labels = one_hot_labels
        self.K = K
        self.P = P
        self.csv=pd.read_csv(csv, names = ['screenshot_path'] + ["class_" + str(i) for i in range(num_classes)])
        for i in range(num_classes):
            self.csv["class_" + str(i)] = self.csv["class_" + str(i)].astype(dtype='float')

        self.img_folder=img_folder
        img_transforms = [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        if ra_num_ops > 0 and ra_magnitude > 0:
            img_transforms = [transforms.RandAugment(ra_num_ops, ra_magnitude)] + img_transforms

        self.img_transforms = transforms.Compose(img_transforms)

        print("total csv rows", len(self.csv.index))
        if id_list_path is not None:
            with open(id_list_path, "r") as f:
                split_ids = set(json.load(f))
            self.csv['split_id'] = self.csv['screenshot_path'].str.replace("\\", "/")
            self.csv = self.csv[self.csv['split_id'].str.contains("/")]
            self.csv['split_id'] = self.csv['split_id'].str.split("/").str[0]
            self.csv = self.csv[self.csv['split_id'].isin(split_ids)]

            self.csv = self.csv.reset_index(drop=True)
            print("filtered csv rows", len(self.csv.index))
            
        if silver_id_list_path_ignores is not None:
            all_ignores = set()
            for ignore_path in silver_id_list_path_ignores:
                with open(ignore_path, "r") as f:
                    all_ignores |= set(json.load(f))
            
            self.csv = self.csv[self.csv['screenshot_path'].str.contains(".")]
            self.csv['split_id'] = self.csv['screenshot_path'].str.split(".").str[0]
            self.csv = self.csv[~self.csv['split_id'].isin(all_ignores)]
            
            self.csv = self.csv.reset_index(drop=True)
            print("filtered csv rows2", len(self.csv.index))
            
        keep_inds = []
        
        for i in range(num_classes):
            keep_inds.extend(list(self.csv.nlargest(K, "class_" + str(i)).index.values))
            
        keep_inds = set(keep_inds)
        df_mat = self.csv[["class_" + str(i) for i in range(num_classes)]]
        
        image_names = []
        image_labels = []
        
        for i in keep_inds:
            if one_hot_labels:
                image_names.append(self.csv.iloc[i]['screenshot_path'])
                image_labels.append(torch.tensor(df_mat.iloc[i].to_numpy()))
            else:
                idxs = np.argsort(df_mat.iloc[i].to_numpy(), axis=-1)[-P:]
                image_name = self.csv.iloc[i]['screenshot_path']
                for idx in idxs:
                    image_names.append(image_name)
                    image_labels.append(idx)
                
        self.image_names = image_names
        self.labels = image_labels
        
        self.class_counter = Counter(self.labels)
        

    #The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self,index):
        index = index % len(self.image_names)
        
        def tryAnother():
            return self.__getitem__(index + 1)
        try:
            img_path = os.path.join(self.img_folder, str(self.image_names[index])).replace("\\", "/")
            image = Image.open(img_path).convert("RGB")

            image=self.img_transforms(image)
            targets=self.labels[index]

            return {'image': image,'label':targets}
        except:
            return tryAnother()

class SilverDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, num_workers=0, silver_id_list_path=None, silver_id_list_path_ignores=None, ra_num_ops=2, ra_magnitude=9, P=1, K=150, silver_csv="../../metadata/screenclassification/silver_webui-multi_topic.csv", img_folder="../../downloads/ds"):
        super(SilverDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        ds1 = EnricoImageDataset(id_list_path="../../metadata/screenclassification/filtered_train_ids.json", one_hot_labels=True, ra_num_ops=ra_num_ops, ra_magnitude=ra_magnitude)
        ds2 = SilverMultilabelImageDataset(csv=silver_csv, img_folder=img_folder, id_list_path=silver_id_list_path, silver_id_list_path_ignores=silver_id_list_path_ignores, P=P, K=K, one_hot_labels=True, ra_num_ops=ra_num_ops, ra_magnitude=ra_magnitude)
        
        combined_ds = CombinedImageDataset([ds1, ds2], [1/15, 14/15])
        
        self.train_dataset = combined_ds
        
        self.val_dataset = EnricoImageDataset(id_list_path="../../metadata/screenclassification/filtered_val_ids.json")
        self.test_dataset = EnricoImageDataset(id_list_path="../../metadata/screenclassification/filtered_test_ids.json")

    def train_dataloader(self):
        # samples_weight = torch.tensor([1 / self.train_dataset.class_counter[t] for t in self.train_dataset.labels])
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
        # return torch.utils.data.DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, sampler=sampler, collate_fn=collate_fn_silver)
        return torch.utils.data.DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=collate_fn_silver_multi)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=collate_fn_silver)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=collate_fn_silver)
    

class EnricoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, num_workers=4, img_size=128, ra_num_ops=-1, ra_magnitude=-1):
        super(EnricoDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = EnricoImageDataset(id_list_path="../../metadata/screenclassification/filtered_train_ids.json", ra_num_ops=ra_num_ops, ra_magnitude=ra_magnitude, img_size=img_size)
        self.val_dataset = EnricoImageDataset(id_list_path="../../metadata/screenclassification/filtered_val_ids.json", img_size=img_size)
        self.test_dataset = EnricoImageDataset(id_list_path="../../metadata/screenclassification/filtered_test_ids.json", img_size=img_size)

    def train_dataloader(self):
        samples_weight = torch.tensor([1 / self.train_dataset.class_counter[t] for t in self.train_dataset.labels])
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
        return torch.utils.data.DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, sampler=sampler, collate_fn=collate_fn_enrico)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=collate_fn_enrico)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=collate_fn_enrico)
    
