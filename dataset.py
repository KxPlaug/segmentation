import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import cv2


class FSS1000(Dataset):
    def __init__(self, root='data', image_set='train', h=512, w=512):
        self.root = root
        self.h = h
        self.w = w
        fss1000_dir = os.path.join(self.root, 'fewshot_data')

        self.classes_name = ['background'] + os.listdir(fss1000_dir)
        self.classes = [0] + list(range(1, len(self.classes_name)))

        if image_set == 'train':
            limits = (1, 6)
        elif image_set == 'val':
            limits = (6, 11)
        self.dataset = []
        for i in range(1, len(self.classes_name)):
            class_dir = os.path.join(fss1000_dir, self.classes_name[i])
            for j in range(limits[0], limits[1]):
                input_path = os.path.join(class_dir, f'{j}.jpg')
                target_path = os.path.join(class_dir, f'{j}.png')
                class_label = self.classes[i]
                self.dataset.append((input_path, target_path, class_label))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_path, target_path, class_label = self.dataset[idx]

        img = np.array(Image.open(input_path))

        if self.h and self.w:
            img = cv2.resize(img, (self.w, self.h), cv2.INTER_NEAREST)

        target = Image.open(target_path)
        target = np.array(target)
        target = np.min(target, axis=2)
        target = target/255*class_label
        if self.h and self.w:
            target = cv2.resize(target, (self.w, self.h), cv2.INTER_NEAREST)

        return torch.Tensor(img).transpose(2, 1).transpose(0, 1), torch.Tensor(target)


class VOCSegmentation(Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
    """

    def __init__(
            self,
            root: str = "data",
            image_set: str = "train",
            h=512,
            w=512
    ):
        super(VOCSegmentation, self).__init__()
        self.year = "2012"
        valid_sets = ["train", "trainval", "val"]

        self.root = root
        self.h = h
        self.w = w

        base_dir = os.path.join('VOCdevkit', 'VOC2012')
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.')

        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = np.array(Image.open(self.images[index]))
        if self.h and self.w:
            img = cv2.resize(img, (self.w, self.h), cv2.INTER_NEAREST)

        target = Image.open(self.masks[index])
        target = np.array(target)
        idx255 = target == np.ones_like(target) * 255
        target[idx255] = 0
        if self.h and self.w:
            target = cv2.resize(target, (self.w, self.h), cv2.INTER_NEAREST)

        return torch.Tensor(img).transpose(2, 1).transpose(0, 1), torch.Tensor(target)

    def __len__(self):
        return len(self.images)


def load_fss(root='data',batch_size=16):
    train_dataloader = torch.utils.data.DataLoader(
        FSS1000(root, 'train'),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(
        FSS1000(root, 'val'),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)
    return train_dataloader, test_dataloader

def load_voc(root='data',batch_size=16):
    train_dataloader = torch.utils.data.DataLoader(
        VOCSegmentation(root, 'train'),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(
        VOCSegmentation(root, 'val'),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)
    return train_dataloader, test_dataloader