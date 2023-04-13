import torch
import os
from PIL import Image
import json
# import orjson
from torchvision import transforms
import pytorch_lightning as pl

import torch.nn.functional as F
from botocore.client import Config
import glob
import random

import zipfile

import xmltodict

# import sys

# sys.path.append('../webuidata')
 
# import download_partial_data_webui

DEVICE_SCALE = {
    "default": 1,
    "iPad-Mini": 2,
    "iPad-Pro": 2,
    "iPhone-13 Pro": 3,
    "iPhone-SE": 3
}

def makeMultiHotVec(idxs, num_classes):
    vec = [1 if i in idxs else 0 for i in range(num_classes)]
    return vec

class VINSUIDataset(torch.utils.data.Dataset):
    def __init__(self, root="../../downloads/vins/All Dataset", class_dict_path = "../../metadata/screenrecognition/class_map_vins_manual.json", id_list_path = "../../metadata/screenrecognition/train_ids_vins.json"):

        with open(id_list_path, "r") as f:
            self.id_list = json.load(f)

        self.root = root
        self.img_transforms = transforms.ToTensor()

        with open(class_dict_path, "r") as f:
            class_dict = json.load(f)

        self.idx2Label = class_dict['idx2Label']
        self.label2Idx = class_dict['label2Idx']
        
    def __len__(self):
        return len(self.id_list)
        
    def __getitem__(self, idx):
        def return_next(): # for debugging
            return VINSUIDataset.__getitem__(self, idx + 1)
        try:
            img_path = os.path.join(self.root, self.id_list[idx])

            pil_img = Image.open(img_path).convert("RGB")
            img = self.img_transforms(pil_img)

            # get bounding box coordinates for each mask

            with open(img_path.replace(".jpg", ".xml").replace("JPEGImages", "Annotations"),"r") as root_file:
                test_dat = root_file.read()

            dd = xmltodict.parse(test_dat)

            boxes = []
            labels = []

            for obj in dd['annotation']['object']:
                bbo = obj['bndbox']
                bb = [float(bbo['xmin']), float(bbo['ymin']), float(bbo['xmax']), float(bbo['ymax'])]
                boxes.append(bb)
                labels.append(self.label2Idx[obj['name']])

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            #iscrowd = torch.zeros((num_labels,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area

            return img, target
        except Exception as e:
            print("failed", idx, self.id_list[idx], str(e))
            return return_next()
    

class VINSUIOneHotLabelDataset(VINSUIDataset):
    def __getitem__(self, idx):
        img, res_dict = super(VINSUIOneHotLabelDataset, self).__getitem__(idx)
        num_classes = 13
        one_hot_labels = F.one_hot(res_dict['labels'], num_classes=num_classes)
        res_dict['labels'] = one_hot_labels
        return img, res_dict

# todo, maybe add more image transformations for data augmentation
class RicoUIDataset(torch.utils.data.Dataset):
    
    
    #os.environ['SM_CHANNEL_TRAINING'] if 'SM_CHANNEL_TRAINING' in os.environ else "/storage/combined"
    
    
    #os.environ['SM_CHANNEL_TRAINING']+'/ui_json' if 'SM_CHANNEL_TRAINING' in os.environ else "/storage/screen_rec/ui_json"
    def __init__(self, root=os.environ['SM_CHANNEL_TRAINING']+'/screenshots' if 'SM_CHANNEL_TRAINING' in os.environ else "/storage/combined", annotationsPath = os.environ['SM_CHANNEL_TRAINING']+'/ui_json' if 'SM_CHANNEL_TRAINING' in os.environ else "/storage/screen_rec/clay_ui_json", ignore_list_path= "ignore_list.json", class_dict_path = "class_dict.json", id_list_path = "train_ids.json"):
        self.root = root
        self.annotationsPath = annotationsPath
        self.img_transforms = transforms.ToTensor()
        # load all image files, sorting them to
        # ensure that they are aligned
        #self.imgs = list(sorted(os.listdir('/storage/test/semantic_annotations'))
        
        with open(id_list_path, "r") as f:
            self.id_list = json.load(f)
        with open(ignore_list_path,"r") as ignore_json:
            ignore_l = json.load(ignore_json)
        self.ignore_list = ignore_l
        
        # self.imgs = [i for i in self.imgs if i[:-4] not in self.ignore_list]
        # self.json_list = [i for i in self.json_list if i[:-5] not in self.ignore_list]
        
        self.id_list = [i for i in self.id_list if i not in self.ignore_list]

        with open(class_dict_path,"r") as classes_json:
            classes_d = json.load(classes_json)
        self.label2Idx= classes_d["label2Idx"]
        self.idx2Label = classes_d["idx2Label"]
        self.num_classes = max([int(k) for k in self.idx2Label.keys()]) + 1
        
    def __len__(self):
        return len(self.id_list)
        
    def __getitem__(self, idx):
        # load images and masks

        # img_path = os.path.join(self.root, self.imgs[idx])
        img_path = os.path.join(self.root, self.id_list[idx] + ".jpg")
        
        img = Image.open(img_path).convert("RGB")
        img = self.img_transforms(img)
          
        

        # get bounding box coordinates for each mask
        
        with open(os.path.join(self.annotationsPath, self.id_list[idx] + ".json"),"r") as root_file:
            root = json.load(root_file)
        json_boxes = root['boxes']
        boxes =[]
        ignore_indices = []

        
        for i in range(len(json_boxes)):
            xmin = json_boxes[i][0]
            ymin = json_boxes[i][1]
            xmax = json_boxes[i][2]
            ymax = json_boxes[i][3]
            
            if xmin >= xmax or ymin >= ymax:
                ignore_indices.append(i)
            boxes.append([xmin, ymin, xmax, ymax])
            
        labels = [self.label2Idx[l] for l in root['labels']]
        
        # filter out the ignored indices
        ignore_indices = set(ignore_indices)
        boxes = [boxes[i] for i in range(len(boxes)) if i not in ignore_indices]
        labels = [labels[i] for i in range(len(labels)) if i not in ignore_indices]
        
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        
        
        # change
        labels = torch.tensor(labels, dtype=torch.long)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        #iscrowd = torch.zeros((num_labels,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area

        return img, target


class RicoUIOneHotLabelDataset(RicoUIDataset):
    def __getitem__(self, idx):
        img, res_dict = super(RicoUIOneHotLabelDataset, self).__getitem__(idx)
        num_classes = self.num_classes
        one_hot_labels = F.one_hot(res_dict['labels'], num_classes=num_classes)
        res_dict['labels'] = one_hot_labels
        return img, res_dict
    
class ClayUIDataset(torch.utils.data.Dataset):
    def __init__(self, root="/combined", annotationsPath =  "/clay_json", ignore_list_path= "ignore_list.json", class_dict_path = "class_dict.json", id_list_path = "train_ids.json"):

        if 'SM_CHANNEL_TRAINING' in os.environ:
            if os.path.exists(os.environ['SM_CHANNEL_TRAINING']+'/clay_dataset.zip'):
                with zipfile.ZipFile(os.environ['SM_CHANNEL_TRAINING']+'/clay_dataset.zip', 'r') as zip_ref:
                    zip_ref.extractall('/')

                os.remove(os.environ['SM_CHANNEL_TRAINING'] + "/clay_dataset.zip")

            if os.path.exists(os.environ['SM_CHANNEL_TRAINING']+'/clay_computed_boxes.zip'):
                with zipfile.ZipFile(os.environ['SM_CHANNEL_TRAINING']+'/clay_computed_boxes.zip', 'r') as zip_ref:
                    zip_ref.extractall('/')

                os.remove(os.environ['SM_CHANNEL_TRAINING'] + "/clay_computed_boxes.zip")
        
        with open(id_list_path, "r") as f:
            self.id_list = json.load(f)

        with open(ignore_list_path,"r") as ignore_json:
            ignore_l = json.load(ignore_json)
        self.ignore_list = ignore_l
        
        self.id_list = [i for i in self.id_list if i not in self.ignore_list]
        
        self.root = root
        self.annotationsPath = annotationsPath
        self.img_transforms = transforms.ToTensor()
        print(os.listdir("/"))
        
    def __len__(self):
        return len(self.id_list)
        
    def __getitem__(self, idx):
        def return_next(): # for debugging
            return ClayUIDataset.__getitem__(self, idx + 1)
        try:
            # img_path = os.path.join(self.root, self.imgs[idx])
            img_path = os.path.join(self.root, self.id_list[idx] + ".jpg")

            pil_img = Image.open(img_path).convert("RGB")
            img = self.img_transforms(pil_img)

            # get bounding box coordinates for each mask

            with open(os.path.join(self.annotationsPath, self.id_list[idx] + ".json"),"r") as root_file:
                test_dict = json.load(root_file)

            # print(test_dict)
            if 'boxes' not in test_dict or 'labels' not in test_dict:
                return return_next()

            rw = test_dict['root_bounds'][2] - test_dict['root_bounds'][0]
            # rh = test_dict['root_bounds'][3] - test_dict['root_bounds'][1]
            rh = pil_img.size[1] * (rw / pil_img.size[0])

            if rw <= 0:
                rw = pil_img.size[0]

            if rh <= 0:
                rh = pil_img.size[1]

            boxes = []
            ignore_indices = []

            for i in range(len(test_dict['boxes'])):
                box = test_dict['boxes'][i]
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                x1 = x1 / rw * pil_img.size[0]
                y1 = y1 / rh * pil_img.size[1]
                x2 = x2 / rw * pil_img.size[0]
                y2 = y2 / rh * pil_img.size[1]

                if x1 >= x2 or y1 >= y2:
                    ignore_indices.append(i)

                boxes.append([x1, y1, x2, y2])

            labels = test_dict['labels']

            for i in range(len(labels)):
                if labels[i] <= 0:
                    ignore_indices.append(i)

            # filter out the ignored indices
            ignore_indices = set(ignore_indices)
            boxes = [boxes[i] for i in range(len(boxes)) if i not in ignore_indices]
            labels = [labels[i] for i in range(len(labels)) if i not in ignore_indices]


            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class


            # change
            labels = torch.tensor(labels, dtype=torch.long)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            #iscrowd = torch.zeros((num_labels,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area

            return img, target
        except Exception as e:
            print("failed", idx, self.id_list[idx], str(e))
            return return_next()
    

class ClayUIOneHotLabelDataset(ClayUIDataset):
    def __getitem__(self, idx):
        img, res_dict = super(ClayUIOneHotLabelDataset, self).__getitem__(idx)
        num_classes = 24
        one_hot_labels = F.one_hot(res_dict['labels'], num_classes=num_classes)
        res_dict['labels'] = one_hot_labels
        return img, res_dict

# todo, maybe add more image transformations for data augmentation
class WebUIPilotDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, root_dir="../../", class_map_file="class_map.json", min_area=10, device_scale=DEVICE_SCALE):
        super(WebUiPilotDataset, self).__init__()
        self.keys = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")]
        # TODO: remove this (only for debugging)
        if data_dir == 'val_dataset':
            self.keys = self.keys[:1000]
        self.root_dir = root_dir
        self.min_area = min_area
        self.device_scale = device_scale
        with open(class_map_file, "r") as f:
            class_map = json.load(f)

        self.idx2Label = class_map['idx2Label']
        self.label2Idx = class_map['label2Idx']
        self.num_classes = max([int(k) for k in self.idx2Label.keys()]) + 1
        self.img_transforms = transforms.ToTensor()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        try:
            idx = idx % len(self.keys)
            key = self.keys[idx]
            with open(key, "r") as f:
                key_dict = json.load(f)
            url_path = os.path.join(self.root_dir, key_dict['key_name']) # path to url.txt file
            url_path = url_path.replace("\\", "/")
            key_filename = url_path.split("/")[-1]
            device_name = "-".join(key_filename.split("-")[:-1])
            img_path = url_path.replace("-url.txt", "-screenshot.png")
            img_pil = Image.open(img_path).convert("RGB")
            img = self.img_transforms(img_pil)
            target = {}
            boxes = []
            labels = []
            for i in range(len(key_dict['labels'])):
                box = key_dict['contentBoxes'][i]
                # skip invalid boxes
                if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
                    continue
                if box[3] <= box[1] or box[2] <= box[0]:
                    continue
                if (box[3] - box[1]) * (box[2] - box[0]) <= self.min_area: # get rid of really small elements
                    continue
                boxes.append(box)
                label = key_dict['labels'][i]
                labelIdx = [self.label2Idx[label[li]] if label[li] in self.label2Idx else self.label2Idx['OTHER'] for li in range(len(label))]
                labelHot = makeMultiHotVec(set(labelIdx), self.num_classes)
                labels.append(labelHot)

            if len(boxes) > 200:
                # print("skipped due to too many objects", len(boxes))
                return self.__getitem__(idx + 1)

            boxes = torch.tensor(boxes, dtype=torch.float)
            boxes *= self.device_scale[device_name]
            
            labels = torch.tensor(labels, dtype=torch.long)

            target['boxes'] = boxes if len(boxes.shape) == 2 else torch.zeros(0, 4)
            target['labels'] = labels
            target['image_id'] = torch.tensor([idx])
            target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes.shape) == 2 else torch.zeros(0)
            target['iscrowd'] = torch.zeros((boxes.shape[0],), dtype=torch.long) if len(boxes.shape) == 2 else torch.zeros(0, dtype=torch.long)

            return img, target # return image and target dict
        except Exception as e:
            print("failed", idx, str(e))
            return self.__getitem__(idx + 1)

        
# todo, maybe add more image transformations for data augmentation
class WebUIDataset(torch.utils.data.Dataset):
    def __init__(self, split_file, boxes_dir='../../downloads/webui-boxes/all_data', rawdata_screenshots_dir='../../downloads/ds', class_map_file="../../metadata/screenrecognition/class_map.json", min_area=100, device_scale=DEVICE_SCALE, max_boxes=100, max_skip_boxes=100):
        super(WebUIDataset, self).__init__()
        self.max_boxes = max_boxes
        self.max_skip_boxes = max_skip_boxes
        self.keys = []
        
        with open(split_file, "r") as f:
            boxes_split = json.load(f)
        
        rawdata_directory = rawdata_screenshots_dir
        for folder in [f for f in os.listdir(boxes_dir) if f in boxes_split]:
            for file in os.listdir(os.path.join(boxes_dir,folder)):
                if os.path.exists(os.path.join(rawdata_directory, folder, file.replace('.json','-screenshot.webp'))):
                    self.keys.append(os.path.join(boxes_dir, folder, file))
        
        self.min_area = min_area
        self.device_scale = device_scale
        with open(class_map_file, "r") as f:
            class_map = json.load(f)
        self.computed_boxes_directory = boxes_dir
        self.rawdata_directory = rawdata_directory
        self.idx2Label = class_map['idx2Label']
        self.label2Idx = class_map['label2Idx']
        self.num_classes = max([int(k) for k in self.idx2Label.keys()]) + 1
        self.img_transforms = transforms.ToTensor()
        

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        try:
            idx = idx % len(self.keys)
            key = self.keys[idx]
            with open(key, "r") as f:
                key_dict = json.load(f)

            img_path = key.replace(".json", "-screenshot.webp")
            img_path = img_path.replace(self.computed_boxes_directory, self.rawdata_directory)

            key_filename = img_path.split("/")[-1]
            device_name = "-".join(key_filename.split("-")[:-1])
            
            
            img_pil = Image.open(img_path).convert("RGB")
            img = self.img_transforms(img_pil)
            target = {}
            boxes = []
            labels = []
            scale = self.device_scale[device_name.split("_")[0]]
            
            inds = list(range(len(key_dict['labels'])))
            random.shuffle(inds)
            
            for i in inds:
                box = key_dict['contentBoxes'][i]
                box[0] *= scale
                box[1] *= scale
                box[2] *= scale
                box[3] *= scale
                
                box[0] = min(max(0, box[0]), img_pil.size[0])
                box[1] = min(max(0, box[1]), img_pil.size[1])
                box[2] = min(max(0, box[2]), img_pil.size[0])
                box[3] = min(max(0, box[3]), img_pil.size[1])
                
                # skip invalid boxes
                if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
                    continue
                if box[3] <= box[1] or box[2] <= box[0]:
                    continue
                if (box[3] - box[1]) * (box[2] - box[0]) <= self.min_area: # get rid of really small elements
                    continue
                boxes.append(box)
                label = key_dict['labels'][i]
                labelIdx = [self.label2Idx[label[li]] if label[li] in self.label2Idx else self.label2Idx['OTHER'] for li in range(len(label))]
                labelHot = makeMultiHotVec(set(labelIdx), self.num_classes)
                labels.append(labelHot)

            if len(boxes) > self.max_skip_boxes:
                # print("skipped due to too many objects", len(boxes))
                return self.__getitem__(idx + 1)

            boxes = torch.tensor(boxes, dtype=torch.float)

            labels = torch.tensor(labels, dtype=torch.long)

            target['boxes'] = boxes if len(boxes.shape) == 2 else torch.zeros(0, 4)
            target['labels'] = labels
            target['image_id'] = torch.tensor([idx])
            target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes.shape) == 2 else torch.zeros(0)
            target['iscrowd'] = torch.zeros((boxes.shape[0],), dtype=torch.long) if len(boxes.shape) == 2 else torch.zeros(0, dtype=torch.long)
            
            for k in target:
                target[k] = target[k][:self.max_boxes]

            return img, target # return image and target dict
        except Exception as e:
            print("failed", idx, str(e))
            return self.__getitem__(idx + 1)

class WebUIPilotDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4, num_workers=2):
        super(WebUIPilotDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = WebUIPilotDataset(data_dir="train_dataset")
        self.val_dataset = WebUiPilotDataset(data_dir="val_dataset")
        self.test_dataset = WebUiPilotDataset(data_dir="test_dataset")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, collate_fn=collate_fn, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, collate_fn=collate_fn, num_workers=self.num_workers, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, collate_fn=collate_fn, num_workers=self.num_workers, batch_size=self.batch_size)
    
class RicoUIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4, num_workers=0, one_hot_labels=True):
        super(RicoUIDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        if one_hot_labels:
            self.train_dataset = RicoUIOneHotLabelDataset(id_list_path="train_ids.json")
            self.val_dataset = RicoUIOneHotLabelDataset(id_list_path="val_ids.json")
            self.test_dataset = RicoUIOneHotLabelDataset(id_list_path="test_ids.json")
        else:
            self.train_dataset = RicoUIDataset(id_list_path="train_ids.json")
            self.val_dataset = RicoUIDataset(id_list_path="val_ids.json")
            self.test_dataset = RicoUIDataset(id_list_path="test_ids.json")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, collate_fn=collate_fn, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, collate_fn=collate_fn, num_workers=self.num_workers, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, collate_fn=collate_fn, num_workers=self.num_workers, batch_size=self.batch_size)

class VINSUIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, num_workers=4, one_hot_labels=True):
        super(VINSUIDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        if one_hot_labels:
            self.train_dataset = VINSUIOneHotLabelDataset(id_list_path="../../metadata/screenrecognition/train_ids_vins.json")
            self.val_dataset = VINSUIOneHotLabelDataset(id_list_path="../../metadata/screenrecognition/val_ids_vins.json")
            self.test_dataset = VINSUIOneHotLabelDataset(id_list_path="../../metadata/screenrecognition/test_ids_vins.json")
        else:
            self.train_dataset = VINSUIDataset(id_list_path="../../metadata/screenrecognition/train_ids_vins.json")
            self.val_dataset = VINSUIDataset(id_list_path="../../metadata/screenrecognition/val_ids_vins.json")
            self.test_dataset = VINSUIDataset(id_list_path="../../metadata/screenrecognition/test_ids_vins.json")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, collate_fn=collate_fn, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, collate_fn=collate_fn, num_workers=self.num_workers, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, collate_fn=collate_fn, num_workers=self.num_workers, batch_size=self.batch_size)

class ClayUIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, num_workers=4, one_hot_labels=True):
        super(ClayUIDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        if one_hot_labels:
            self.train_dataset = ClayUIOneHotLabelDataset(id_list_path="train_ids.json")
            self.val_dataset = ClayUIOneHotLabelDataset(id_list_path="val_ids.json")
            self.test_dataset = ClayUIOneHotLabelDataset(id_list_path="test_ids.json")
        else:
            self.train_dataset = ClayUIDataset(id_list_path="train_ids.json")
            self.val_dataset = ClayUIDataset(id_list_path="val_ids.json")
            self.test_dataset = ClayUIDataset(id_list_path="test_ids.json")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, collate_fn=collate_fn, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, collate_fn=collate_fn, num_workers=self.num_workers, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, collate_fn=collate_fn, num_workers=self.num_workers, batch_size=self.batch_size)
    
    

# https://github.com/pytorch/vision/blob/5985504cc32011fbd4312600b4492d8ae0dd13b4/references/detection/utils.py#L203
def collate_fn(batch):
    return tuple(zip(*batch))

class WebUIDataModule(pl.LightningDataModule):
    def __init__(self, train_split_file, val_split_file="../../downloads/val_split_webui.json", test_split_file="../../downloads/test_split_webui.json", batch_size=8, num_workers=4):
        super(WebUIDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = WebUIDataset(split_file = train_split_file)
        self.val_dataset = WebUIDataset(split_file = val_split_file)
        self.test_dataset = WebUIDataset(split_file = test_split_file)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, collate_fn=collate_fn, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, collate_fn=collate_fn, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True) # shuffle so that we can eval on subset

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, collate_fn=collate_fn, num_workers=self.num_workers, batch_size=self.batch_size)
