import os
import sys
import torch
import torch.nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from networks.resnet import resnet50


model = resnet50(num_classes=1)
state_dict = torch.load(opt.model_path, map_location='cpu')
model.load_state_dict(state_dict['model'])
if(not opt.use_cpu):
  model.cuda()
model.eval()

# Transform
trans_init = []
if(opt.crop is not None):
  trans_init = [transforms.CenterCrop(opt.crop),]
  print('Cropping to [%i]'%opt.crop)
else:
  print('Not cropping')
trans = transforms.Compose(trans_init + [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = trans(Image.open(opt.file).convert('RGB'))

with torch.no_grad():
    in_tens = img.unsqueeze(0)
    if(not opt.use_cpu):
    	in_tens = in_tens.cuda()
    prob = model(in_tens).sigmoid().item()

print('probability of being synthetic: {:.2f}%'.format(prob * 100))