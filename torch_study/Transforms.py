from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2

#transforms的使用
img_path="data//train//ants_image//0013035.jpg"
img=Image.open(img_path)
# print(img)
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)

writer=SummaryWriter("logs")
writer.add_image("Tensor_img",tensor_img)
writer.close()
# cv_img=cv2.imread(img_path)
print(tensor_img)
