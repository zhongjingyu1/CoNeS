import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse
from utils.pml_data import generate_uniform_cv_candidate_labels
import numpy as np
import torch
import torch.utils.data as data
from utils.util import *

object_categories = ['clouds', 'sky', 'person', 'street', 'window', 'tattoo', 'wedding', 'animal', 'cat', 'buildings',
                     'tree', 'airport', 'plane', 'water', 'grass', 'cars', 'road', 'snow', 'sunset', 'railroad',
                     'train', 'flowers', 'plants', 'house', 'military', 'horses', 'nighttime', 'lake', 'rocks',
                     'waterfall', 'sun', 'vehicle', 'sports', 'reflection', 'temple', 'statue', 'ocean', 'town',
                     'beach', 'tower', 'toy', 'book', 'bridge', 'fire', 'mountain', 'rainbow', 'garden', 'police',
                     'coral', 'fox', 'sign', 'dog', 'cityscape', 'sand', 'dancing', 'leaf', 'tiger', 'moon', 'birds',
                     'food', 'cow', 'valley', 'fish', 'harbor', 'bear', 'castle', 'boats', 'running', 'glacier',
                     'swimmers', 'elk', 'frost', 'protest', 'soccer', 'flags', 'zebra', 'surf', 'whales', 'computer',
                     'earthquake', 'map']
object_categories_map = {'clouds': 0, 'sky': 1, 'person': 2, 'street': 3, 'window': 4, 'tattoo': 5, 'wedding': 6,
                         'animal': 7, 'cat': 8, 'buildings': 9, 'tree': 10, 'airport': 11, 'plane': 12, 'water': 13,
                         'grass': 14, 'cars': 15, 'road': 16, 'snow': 17, 'sunset': 18, 'railroad': 19, 'train': 20,
                         'flowers': 21, 'plants': 22, 'house': 23, 'military': 24, 'horses': 25, 'nighttime': 26,
                         'lake': 27, 'rocks': 28, 'waterfall': 29, 'sun': 30, 'vehicle': 31, 'sports': 32,
                         'reflection': 33, 'temple': 34, 'statue': 35, 'ocean': 36, 'town': 37, 'beach': 38,
                         'tower': 39, 'toy': 40, 'book': 41, 'bridge': 42, 'fire': 43, 'mountain': 44, 'rainbow': 45,
                         'garden': 46, 'police': 47, 'coral': 48, 'fox': 49, 'sign': 50, 'dog': 51, 'cityscape': 52,
                         'sand': 53, 'dancing': 54, 'leaf': 55, 'tiger': 56, 'moon': 57, 'birds': 58, 'food': 59,
                         'cow': 60, 'valley': 61, 'fish': 62, 'harbor': 63, 'bear': 64, 'castle': 65, 'boats': 66,
                         'running': 67, 'glacier': 68, 'swimmers': 69, 'elk': 70, 'frost': 71, 'protest': 72,
                         'soccer': 73, 'flags': 74, 'zebra': 75, 'surf': 76, 'whales': 77, 'computer': 78,
                         'earthquake': 79, 'map': 80}


def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
            # data.append([name, label])
            # print('%s  %d' % (name, label))
    return data


def read_object_labels_csv(file, set, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        adj = np.zeros((len(object_categories), len(object_categories)))
        for row in reader:
            if row[2] == set:
                if header and rownum == 0:
                    header = row
                else:
                    if num_categories == 0:
                        num_categories = len(row) - 1
                    name = row[0]
                    label = []
                    label_str = row[1][1:-1]
                    label_str_list = label_str.split(', ')
                    for l in label_str_list:
                        cur = l[1:-1]
                        label.append(object_categories_map[cur])
                    labels = (np.asarray(label)).astype(np.float32)
                    # Create adjcent matrix here
                    # for i in labels:
                    #     for j in labels:
                    #         adj[int(i)][int(j)] += 1
                    targets = [-1] * len(object_categories)
                    for l in labels:
                        targets[int(l)] = 1
                    targets = torch.tensor(targets)
                    item = (name, targets)
                    images.append(item)

                rownum += 1
    return images

class NusWideClassification(data.Dataset):
    def __init__(self, dataset, set, path_images, transform=None, target_transform=None, adj=None):
        self.dataset = dataset
        self.path_images = path_images
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        # define path of csv file
        # define filename of csv file
        file_csv = './data/nuswide/nus_wid_data.csv'

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv, set)

        self.true_labels = []
        for an in self.images:
            path, target = an
            target[target == -1] = 0
            self.true_labels.append(target.tolist())
        self.true_labels = np.array(self.true_labels)
        self.Partial = torch.Tensor(self.true_labels)

        print('[dataset] Nus-Wide classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        target[target == -1] = 0
        img = Image.open(os.path.join(self.path_images, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        message = {
            "img_path": path,
            "target": target.float(),
            "img": img,
            "index_num": index
        }
        return message

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)

class NusWideClassification_partial(data.Dataset):
    def __init__(self, dataset, set, path_images, partial_rate, transform=None, target_transform=None):
        self.path_images = path_images
        self.set = set
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset

        file_csv = './data/nuswide/nus_wid_data.csv'

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv, set)

        self.true_labels = []
        for an in self.images:
            path, target = an
            target[target == -1] = 0
            self.true_labels.append(target.tolist())
        self.true_labels = np.array(self.true_labels)

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

        self.Partial = torch.Tensor(Partial_Labels)


    def __getitem__(self, index):
        path, _ = self.images[index]
        img = Image.open(os.path.join(self.path_images, path)).convert('RGB')
        target = self.Partial[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        message = {
            "img_path": path,
            "target": target.float(),
            "img": img,
            "index_num": index
        }
        return message

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)