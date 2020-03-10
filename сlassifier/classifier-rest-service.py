# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import glob
from PIL import Image
from torch.autograd import Variable
from scipy.special import softmax


import requests
import json
import base64
import sys

import cv2
import io
import logging

import tornado.ioloop
import tornado.gen
import tornado.web
import tornado.options

nas_root_path = "/mnt/qnas/images"
port = 8921

log_file_name = 'furniture-classifier-rest-service.log'
url = r"/furniture-classifier/image"

logger = logging.getLogger('furniture-classifier-rest-service')
logger.setLevel(logging.DEBUG)
logger.propagate = False
# create file handler which logs even debug messages
fh = logging.FileHandler(log_file_name)
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

weights_path = '/net/qnasimg/FURNITURE/okhrimenko/furniture_classifier/files#CL_ALL_v5/snapshot_32.pth.tar'
input_size = 224

gpu_id = "cuda:0"
device = torch.device(gpu_id)
############## RUN ##############
checkpoint = torch.load(weights_path, map_location=gpu_id)
    
class_to_idx = checkpoint['class_to_idx']
idx_to_class = {val: key for key, val in class_to_idx.items()}
num_classes = len(class_to_idx)

# Initialize the model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)
model.load_state_dict(checkpoint['state_dict'])

############## DATA ##############
data_transform = transforms.Compose([
transforms.Resize((input_size, input_size)),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class MainHandler(tornado.web.RequestHandler):      # определяем обработчики запросов handlers
    logger.info('Web-service loaded successfully.')
    def post(self):
        try:
            logger.info("Start POST processing.")
            json_str = json.loads(self.request.body)
            if 'imageBase64' in json_str:
                base64Data = json_str["imageBase64"]
                missing_padding = len(base64Data) % 4
                if missing_padding != 0:
                    base64Data += b'=' * (4 - missing_padding)

                imgdata = base64.b64decode(base64Data)
                image = Image.open(io.BytesIO(imgdata))

            if "imagePath" in json_str:
                imagePath = json_str["imagePath"]
                imagePath = nas_root_path + imagePath
                logger.info("Image path=" + imagePath)
                # Load the image
                f = open(imagePath, "rb")
                image = Image.open(f)

            img = Image.fromarray(np.array(image))
            img = img.convert('RGB')            
            
            image_tensor = data_transform(img).float()
            image_tensor = image_tensor.unsqueeze_(0)
            inpt = Variable(image_tensor)
            inpt = inpt.to(device)
    
            with torch.no_grad():
                model.eval()

                output = model(inpt)
                score = float(softmax(output.data.cpu().numpy()).max())
                index = output.data.cpu().numpy().argmax()
                
                logger.info ('Class: {:s}, score: {:}'.format(idx_to_class[index], score))
                res = {"detectedClasses": [{"label": idx_to_class[index], "score": score}]}
            self.write(res)
        except:
            logger.exception("Exception in POST handler.")
            raise

def make_app():

    return tornado.web.Application([
        (url, MainHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(port)
    tornado.ioloop.IOLoop.current().start()



