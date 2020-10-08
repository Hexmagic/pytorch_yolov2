import os
import argparse
from os import stat
import time
import torch
from torch.autograd import Variable
from PIL import Image

from yolov2 import YOLOv2
import numpy as np
from visualize import draw_detection_boxes
import matplotlib.pyplot as plt
from yolo_eval import yolo_eval
import torch.nn.functional as F

def prepare_im_data(img):
    """
    Prepare image data that will be feed to network.

    Arguments:
    img -- PIL.Image object

    Returns:
    im_data -- tensor of shape (3, H, W).
    im_info -- dictionary {height, width}

    """

    im_info = dict()
    im_info['width'], im_info['height'] = img.size

    # resize the image
    H, W = (416, 416)
    im_data = img.resize((H, W))

    # to torch tensor
    im_data = torch.from_numpy(np.array(im_data)).float() / 255

    im_data = im_data.permute(2, 0, 1).unsqueeze(0)

    return im_data, im_info


def parse_args():

    parser = argparse.ArgumentParser('Yolo v2')
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output', type=str)
    parser.add_argument('--model_name', dest='model_name',
                        default='yolov2_160.pth', type=str)

    args = parser.parse_args()
    return args


def demo():
    args = parse_args()
    print('call with args: {}'.format(args))

    # input images
    images_dir = 'images'
    images_names = ['image1.jpg', 'image2.jpg']

    classes = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')

    #model = Yolov2()
    #state = torch.load('output/yolov2_epoch_160.pth')['model']
    # model.load_state_dict(state)

    model = torch.load('weights/yolov2_155.pth')
    print('loaded')

    # model_path = os.path.join(args.output_dir, args.model_name + '.pth')
    # print('loading model from {}'.format(model_path))
    # if torch.cuda.is_available():
    #     checkpoint = torch.load(model_path)
    # else:
    #     checkpoint = torch.load(model_path, map_location='cpu')
    # model.load_state_dict(checkpoint['model'])

    # if args.use_cuda:
    model.cuda()

    model.eval()
    print('model loaded')

    for image_name in images_names:
        image_path = os.path.join(images_dir, image_name)
        img = Image.open(image_path)
        im_data, im_info = prepare_im_data(img)
        im_data_variable = Variable(im_data).cuda()

        tic = time.time()

        output = model(im_data_variable)
        B, C, H, W = output.size()
        out = output.permute(0, 2, 3,
                             1).contiguous().view(B, H * W * 5, 5 + 20)

        xy_pred = torch.sigmoid(out[:, :, 0:2])
        conf_pred = torch.sigmoid(out[:, :, 4:5])
        hw_pred = torch.exp(out[:, :, 2:4])
        class_score = out[:, :, 5:]
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)
        yolo_output = [delta_pred, conf_pred, class_pred]
        yolo_output = [item[0].data for item in yolo_output]
        detections = yolo_eval(yolo_output, im_info,
                               conf_threshold=0.5, nms_threshold=0.4)

        toc = time.time()
        cost_time = toc - tic
        print('im detect, cost time {:4f}, FPS: {}'.format(
            toc-tic, int(1 / cost_time)))

        det_boxes = detections[:, :5].cpu().numpy()
        det_classes = detections[:, -1].long().cpu().numpy()
        im2show = draw_detection_boxes(
            img, det_boxes, det_classes, class_names=classes)
        plt.figure()
        plt.imshow(im2show)
        plt.show()


if __name__ == '__main__':
    demo()
