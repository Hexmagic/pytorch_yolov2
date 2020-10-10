import os
import pdb
from sys import platform

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from numpy.lib.type_check import imag
from PIL import Image
import xml.etree.ElementTree as ET
from torch.nn import functional as F
from torchvision.transforms import ToTensor

from util.augmentation import augment_img


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


class VOCDataset(torch.utils.data.Dataset):
    class_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                   'train', 'tvmonitor')
    sep = '\\' if platform == 'win32' else '/'

    def __init__(self,
                 data_dir,
                 split='trainval',
                 transform=None,
                 img_size=416,
                 rtn_path=False,
                 multi_scale=False,
                 keep_difficult=False):
        """Dataset for VOC data.
                Args:
                        data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
                """
        self.rtn_path = rtn_path
        self.multi_scale = multi_scale
        self.data_dir = data_dir
        self.split = split
        if split != 'test':
            image_sets_file = [
                os.path.join(self.data_dir, f'VOC{year}', "ImageSets", "Main",
                             "%s.txt" % self.split) for year in [2007, 2012]
            ]
            self.ids = VOCDataset._read_image_ids(image_sets_file)
        else:
            image_sets_file = [
                os.path.join(self.data_dir, f'VOC{year}', "ImageSets", "Main",
                             "%s.txt" % self.split) for year in [2007, 2012]
            ]
            self.ids = VOCDataset._read_image_ids(image_sets_file)
            #import pdb; pdb.set_trace()
        self.keep_difficult = keep_difficult
        self.batch_count = 0
        self.img_size = 416
        self.class_dict = {
            class_name: i
            for i, class_name in enumerate(self.class_names)
        }

    def resize(self, img, size):
        image = F.interpolate(img.unsqueeze(0), size=size,
                              mode="nearest").squeeze(0)
        return image

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        w, h = image.size
        image, boxes, labels = augment_img(image, boxes, labels)
        boxes[:, 0::2] = np.clip(boxes[:, 0::2] / w, 0.001, 0.999)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2] / h, 0.001, 0.999)
        image = image.resize((416, 416))
        image = torch.from_numpy(np.array(image)).float() / 255
        image = image.permute(2, 0, 1)
        boxes = torch.from_numpy(boxes)
        return image, boxes, torch.from_numpy(labels), torch.Tensor([len(boxes)]).long()

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_files):
        ids = []
        for filename in image_sets_files:
            with open(filename) as f:
                lst = filename.split(VOCDataset.sep)
                lst = lst[:-1]
                lst[2] = 'Annotations'
                for line in f:
                    lst[3] = f'{line.strip()}.xml'
                    ids.append(VOCDataset.sep.join(lst))
        return ids

    def _get_annotation(self, image_id):
        annotation_file = image_id
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(
                int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes,
                         dtype=np.float32), np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def get_img_info(self, index):
        img_id = self.ids[index]
        annotation_file = os.path.join(self.data_dir, "Annotations",
                                       "%s.xml" % img_id)
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(
            map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def _read_image(self, image_id):
        lst = image_id.split(VOCDataset.sep)
        lst[2] = 'JPEGImages'
        lst[3] = lst[3].replace('.xml', '.jpg')
        image_file = VOCDataset.sep.join(lst)
        image = Image.open(image_file).convert("RGB")
        #image = np.array(image)
        return image


def detection_collate(batch):
    """
    Collate data of different batch, it is because the boxes and gt_classes have changeable length.
    This function will pad the boxes and gt_classes with zero.

    Arguments:
    batch -- list of tuple (im, boxes, gt_classes)

    im_data -- tensor of shape (3, H, W)
    boxes -- tensor of shape (N, 4)
    gt_classes -- tensor of shape (N)
    num_obj -- tensor of shape (1)

    Returns:

    tuple
    1) tensor of shape (batch_size, 3, H, W)
    2) tensor of shape (batch_size, N, 4)
    3) tensor of shape (batch_size, N)
    4) tensor of shape (batch_size, 1)

    """

    # kind of hack, this will break down a list of tuple into
    # individual list
    bsize = len(batch)
    im_data, boxes, gt_classes, num_obj = zip(*batch)
    max_num_obj = max([x.item() for x in num_obj])
    padded_boxes = torch.zeros((bsize, max_num_obj, 4))
    padded_classes = torch.zeros((bsize, max_num_obj,))

    for i in range(bsize):
        padded_boxes[i, :num_obj[i].item(), :] = boxes[i]
        padded_classes[i, :num_obj[i].item()] = gt_classes[i]

    return torch.stack(im_data, 0), padded_boxes, padded_classes, torch.stack(num_obj, 0)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = VOCDataset(data_dir='VOCdevkit', split='train')
    load = DataLoader(dataset,
                      batch_size=2,
                      collate_fn=detection_collate,
                      shuffle=True)
    i = 0
    for ele in load:
        i += 1
        if i > 100:
            break
        print(ele[0].shape)
        print(ele[1].shape)
