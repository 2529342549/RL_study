import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from model import *

image_path = '../../images/cat.png'
image = Image.open(image_path)
print(image)
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

model = torch.load("../../model/model_9.pth")
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image.cuda())

print(output)
print(output.argmax(1))
