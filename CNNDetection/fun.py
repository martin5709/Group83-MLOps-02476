import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image

# Code adapted from CCNDetect

def get_synth_prob(model, path, device):
    trans_init = []
    trans = transforms.Compose(trans_init + [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = trans(Image.open(path).convert('RGB'))
    model.eval()
    with torch.no_grad():
        in_tens = img.unsqueeze(0).to(device)
        prob = model(in_tens).sigmoid().item()

    # print('probability of being synthetic: {:.2f}%'.format(prob * 100))
    return prob