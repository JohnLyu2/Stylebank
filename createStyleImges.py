import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision import transforms

import utils
from networks import StyleBankNet
from torchvision.utils import save_image
from PIL import Image

SAVE_DIR = "caltech_decode"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content_test_dataset = datasets.ImageFolder(root="caltech_101_testing", transform=utils.content_img_transform)
content_test_dataloader = torch.utils.data.DataLoader(content_test_dataset, batch_size=1, num_workers=1)

model = StyleBankNet(8).to(device)

model_weight_path = "weights_cc_0410/model.pth"

if os.path.exists(model_weight_path):
    model_state = torch.load(model_weight_path, map_location=device)
    model.load_state_dict(model_state)
    model.to(device)
else:
    raise Exception('cannot find model weights')

for imgs, labels in content_test_dataloader:
    decode_imgs = model(imgs.to(device))
    decode_img = decode_imgs.squeeze(0)
    # utils.showimg(decode_img.detach())
    decode_img = transforms.ToPILImage()(decode_img.to(device))
    label_name = content_test_dataset.classes[labels[0].item()]
    # save the image to the decode directory within the subfolder of the class
    decode_dir = os.path.join(SAVE_DIR, label_name)
    if not os.path.exists(decode_dir):
        os.makedirs(decode_dir)
    decode_img_path = os.path.join(decode_dir, f"{time.time()}.jpg")
    decode_img.save(decode_img_path)