#!/usr/bin/env python3
# This script is adapted from https://github.com/biubug6/Pytorch_Retinaface 

import os
import torch
import torch.backends.cudnn as cudnn

from retinaface.models.retinaface import RetinaFace
from retinaface.utils.box_utils import decode
from retinaface.utils.timer import Timer
from retinaface.data import cfg_re50
from retinaface.layers.functions.prior_box import PriorBox
from retinaface.utils.nms.py_cpu_nms import py_cpu_nms

from retinaface_utils import * 

import numpy as np
import cv2


def face_detector(img, out_name, net, save_image=True):
    torch.set_grad_enabled(False)
    cfg = cfg_re50
    origin_size = True
    confidence_threshold = 0.02
    nms_threshold = 0.4
    vis_thres = 0.98
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load img in 
    img_raw = cv2.imread(img, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)

    # testing scale
    target_size = 1600
    max_size = 2150
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    resize = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    if origin_size:
        resize = 1

    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, _ = net(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)

    dets = dets[keep, :]

    # save image
    if save_image:
        for b in dets:
            if b[4] < vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # save image
        if not os.path.exists("./results/"):
            os.makedirs("./results/")
        name = "./results/" + str(out_name) + ".jpg"
        cv2.imwrite(name, img_raw)
    results = []
    for face in dets:
        if face[4] > vis_thres:
            results.append(face) 

    return results # returns the bounding boxes 
