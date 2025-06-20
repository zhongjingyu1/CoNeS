import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import torch
import numpy as np
from utils.pml_data import generate_uniform_cv_candidate_labels
import os

class DataSet_Partial(Dataset):
    def __init__(self,
                ann_files,
                augs,
                img_size,
                dataset,
                partial_rate,):
        self.dataset = dataset
        self.ann_files = ann_files
        self.augment = self.augs_function(augs, img_size)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
            ] 
            # In this paper, we normalize the image data to [0, 1]
            # You can also use the so called 'ImageNet' Normalization method
        )
        self.anns = []
        self.load_anns()
        print(self.augment)

        self.true_labels = []
        for an in self.anns:
            labels = an['target']
            self.true_labels.append(labels)
        self.true_labels=np.array(self.true_labels)

        data_dir_prod = os.path.join('pre-processed-data')
        if not os.path.exists(data_dir_prod):
            os.makedirs(data_dir_prod)
        print('==> Loading local data copy in the partial multi-label setup')
        data_file = "{ds}_{pr}.npy".format(
            ds=dataset,
            pr=partial_rate)

        save_path = os.path.join(data_dir_prod, data_file)
        if not os.path.exists(save_path):
            Partial_Labels = generate_uniform_cv_candidate_labels(self.true_labels, partial_rate)
            data_dict = {
                'partial_labels': Partial_Labels
            }
            save_path = os.path.join(data_dir_prod, data_file)
            with open(save_path, 'wb') as f:
                np.save(f, data_dict)
            print('local data saved at ', save_path)
        else:
            data_dict = np.load(save_path, allow_pickle=True).item()
            Partial_Labels = data_dict['partial_labels']

        Partial_Labels_1 = Partial_Labels.tolist()
        for i in range(Partial_Labels.shape[0]):
            self.anns[i]['target'] = Partial_Labels_1[i]

        self.Partial = torch.Tensor(Partial_Labels)
        # in wider dataset we use vit models
        # so transformation has been changed
        if self.dataset == "wider":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ] 
            )        

    def augs_function(self, augs, img_size):            
        t = []
        if 'randomflip' in augs:
            t.append(transforms.RandomHorizontalFlip())
        if 'ColorJitter' in augs:
            t.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0))
        if 'resizedcrop' in augs:
            t.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)))
        if 'RandAugment' in augs:
            t.append(RandAugment())

        t.append(transforms.Resize((img_size, img_size)))

        return transforms.Compose(t)
    
    def load_anns(self):
        self.anns = []
        for ann_file in self.ann_files:
            json_data = json.load(open(ann_file, "r"))
            self.anns += json_data

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        idx = idx % len(self)
        ann = self.anns[idx]
        img = Image.open(ann["img_path"]).convert("RGB")

        if self.dataset == "wider":
            x, y, w, h = ann['bbox']
            img_area = img.crop([x, y, x+w, y+h])
            img_area = self.augment(img_area)
            img_area = self.transform(img_area)
            message = {
                "img_path": ann['img_path'],
                "target": torch.Tensor(ann['target']),
                "img": img_area,
                "index_num": idx
            }
        else: # voc and coco
            img = self.augment(img)
            img = self.transform(img)
            message = {
                "img_path": ann["img_path"],
                "target": torch.Tensor(ann["target"]),
                "img": img,
                "index_num": idx
            }

        return message
