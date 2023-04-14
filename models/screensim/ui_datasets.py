import torch
import json
import random
import os
from PIL import Image
from tqdm import tqdm
from random import choices
from torchvision import transforms
import pytorch_lightning as pl

def random_viewport_from_full(height, w, h):
    h1 = int(random.random() * (h - height))
    h2 = h1 + height
    viewport = (0, h1, w, h2)
    return viewport

def random_viewport_pair_from_full(img_full, height_ratio):
    # print(img_full)
    img_pil = Image.open(img_full).convert("RGB")
    w, h = img_pil.size
    height = int(w * height_ratio)
    viewport1 = random_viewport_from_full(height, w, h)
    vh1 = viewport1[1]
    delta = int(random.random() * (2 * height)) - height
    vh2 = vh1 + delta
    vh2 = min(max(0, vh2), h - height)
    viewport2 = (0, vh2, w, vh2 + height)
    view1 = img_pil.crop(viewport1)
    view2 = img_pil.copy().crop(viewport2)
    return (view1, view2)

class WebUISimilarityDataset(torch.utils.data.IterableDataset):
    def __init__(self, split_file="../../downloads/train_split_web350k.json", root_dir="../../downloads/ds", domain_map_file="../../metadata/screensim/domain_map.json", duplicate_map_file="../../metadata/screensim/duplicate_map.json", device_name="iPhone-13 Pro", scroll_height_ratio=2.164, img_size=(256, 128), uda_dir="../../downloads/rico/combined", uda_ignore_id_files=["../../metadata/screenclassification/filtered_train_ids.json", "../../metadata/screenclassification/filtered_val_ids.json", "../../metadata/screenclassification/filtered_test_ids.json"]):
        super(WebUISimilarityDataset, self).__init__()
        
        self.root_dir = root_dir
        self.device_name = device_name
        self.scroll_height_ratio = scroll_height_ratio
        
        # filter by split file
        with open(split_file, "r") as f:
            split_list = json.load(f)
        split_set = set([str(s) for s in split_list])
        
        with open(domain_map_file, "r") as f:
            self.domain_map = json.load(f)
            
        self.domain_list = []
        for dn in tqdm(self.domain_map):
            if all([url[1] in split_set for url in self.domain_map[dn]]) and len(set([u[0] for u in self.domain_map[dn]])) > 1:
                self.domain_list.append(dn)
            
        with open(duplicate_map_file, "r") as f:
            self.duplicate_map = json.load(f)
            
        self.duplicate_list = []
        for dn in tqdm(self.duplicate_map):
            if all([url in split_set for url in self.duplicate_map[dn]]):
                self.duplicate_list.append(dn)
                
        self.img_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        ignore_ids = set()
        for ignore_file in uda_ignore_id_files:
            with open(ignore_file, "r") as f:
                ignore_file_ids = set(json.load(f))
                ignore_ids |= ignore_file_ids
        
        self.uda_dir = uda_dir
        self.uda_files = [f for f in os.listdir(uda_dir) if f.endswith(".jpg") and f.replace(".jpg", "") not in ignore_ids]
   
    def sample_same_scroll(self):
        try:
            # randomly choose a domain
            random_domain = random.choice(self.domain_list)
            # randomly choose a crawl id from the domain
            domain_urls = self.domain_map[random_domain]
            crawl_id = random.choice(domain_urls)[1]
            screenshot_full_path = os.path.join(self.root_dir, crawl_id, self.device_name + "-screenshot-full.webp")
            pil_img1, pil_img2 = random_viewport_pair_from_full(screenshot_full_path, self.scroll_height_ratio)
            return pil_img1, pil_img2
        except:
            return self.sample_same_scroll()

    def sample_same_screen(self):
        try:
            # randomly choose a duplicated URL
            random_duplicate = random.choice(self.duplicate_list)
            sampled_screens = random.sample(self.duplicate_map[random_duplicate], 2)
            img1_path = os.path.join(self.root_dir, sampled_screens[0], self.device_name + "-screenshot.webp")
            img2_path = os.path.join(self.root_dir, sampled_screens[1], self.device_name + "-screenshot.webp")
            pil_img1 = Image.open(img1_path).convert("RGB")
            pil_img2 = Image.open(img2_path).convert("RGB")
            return pil_img1, pil_img2
        except:
            return self.sample_same_screen()

    def sample_same_domain(self): # same domain but not same path
        try:
            random_domain = random.choice(self.domain_list)
            url1 = random.choice(self.domain_map[random_domain])
            candidates = [u for u in self.domain_map[random_domain] if u[0] != url1[0]]
            url2 = random.choice(candidates)
            img1_path = os.path.join(self.root_dir, url1[1], self.device_name + "-screenshot.webp")
            img2_path = os.path.join(self.root_dir, url2[1], self.device_name + "-screenshot.webp")
            pil_img1 = Image.open(img1_path).convert("RGB")
            pil_img2 = Image.open(img2_path).convert("RGB")
            return pil_img1, pil_img2
        except:
            return self.sample_same_domain()

    def sample_different_domain(self): # different domain
        try:
            sampled_domains = random.sample(self.domain_list, 2)
            domain1 = sampled_domains[0]
            domain2 = sampled_domains[1]
            url1 = random.choice(self.domain_map[domain1])
            url2 = random.choice(self.domain_map[domain2])
            img1_path = os.path.join(self.root_dir, url1[1], self.device_name + "-screenshot.webp")
            img2_path = os.path.join(self.root_dir, url2[1], self.device_name + "-screenshot.webp")
            pil_img1 = Image.open(img1_path).convert("RGB")
            pil_img2 = Image.open(img2_path).convert("RGB")
            return pil_img1, pil_img2
        except:
            return self.sample_different_domain()
        
    def sample_uda_img(self):
        try:
            img_path = os.path.join(self.uda_dir, random.choice(self.uda_files))
            return Image.open(img_path).convert("RGB")
        except:
            return self.sample_uda_img()
    
    def __iter__(self):
        while True:
            probs = [0.25, 0.25, 0.25, 0.25]
            funcs = [self.sample_same_scroll, self.sample_same_screen, self.sample_same_domain, self.sample_different_domain]
            si = choices(list(range(len(funcs))), probs)[0]
            func = funcs[si]
            res = func()
            label = si < 2 # first two are considered same screen
            
            yield {'label': label,
                'image1': self.img_transforms(res[0]),
                'image2': self.img_transforms(res[1]),
                'imageuda1': self.img_transforms(self.sample_uda_img()),
                'imageuda2': self.img_transforms(self.sample_uda_img())}
            

class WebUISimilarityDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, num_workers=4, split_file="../../downloads/train_split_web350k.json", root_dir="../../downloads/ds", domain_map_file="../../metadata/screensim/domain_map.json", duplicate_map_file="../../metadata/screensim/duplicate_map.json", device_name="iPhone-13 Pro", scroll_height_ratio=2.164, img_size=128):
        super(WebUISimilarityDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_file = split_file
        
        self.train_dataset = WebUISimilarityDataset(split_file=split_file)
        self.val_dataset = WebUISimilarityDataset(split_file="../../downloads/val_split_webui.json")
        self.test_dataset = WebUISimilarityDataset(split_file="../../downloads/test_split_webui.json")
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size)
