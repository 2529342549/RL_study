from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("images/6240338_93729615ec.jpg")
# toTensor使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize使用
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([5, 2, 3], [2, 4, 5])
img_norm = trans_norm(img_tensor)
print(img_tensor[0][0][0])
writer.add_image("Normalize", img_norm, 2)

# Resize使用
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL ->resize -> img_resize: PIL
img_resize = trans_resize(img)
# img_resize PIL ->totensor ->img_resize: tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose - resize -2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Compose", img_resize_2, 1)
writer.close()
# print(img)
