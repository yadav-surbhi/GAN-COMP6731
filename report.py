import json
import math
import os
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

import Generator as gen
import torchvision.utils as vutils
import DataManager as dm

model_type = 'CIFAR'
# model_type = 'MNIST'
print('Specify the Dataset:MNIST or CIFAR')
model_type =input()
while model_type not in ["MNIST","CIFAR"]:
    print('Specify the Dataset:MNIST or CIFAR')
    model_type =input()

MODELS_PATH = os.getcwd() + "/Models/"
IMAGES_FOLDER = os.getcwd() + "/Images"
model_name = model_type + "_model"

model_path = MODELS_PATH + model_name


def plot_history(d_loss, g_loss):
    plt.plot(g_loss, label='g loss', c='tab:orange')
    plt.plot(d_loss, label='d loss', c='tab:blue')
    plt.legend()
    # save plot to file
    plt.savefig(IMAGES_FOLDER + '/' + model_type + '_plot_loss.png')
    plt.close()


try:
    with open(MODELS_PATH + model_type + '_loss.json', 'r') as filehandle:
        loss = json.load(filehandle)
        d_loss = loss[0]
        g_loss = loss[1]
except FileNotFoundError:
    print("No saved loss data file.")

plot_history(d_loss, g_loss)


# Example of using the latest trained model
# create a generator instance
generator = gen.getGenerator(model_type)
# load the trained model weights
checkpoint = torch.load(MODELS_PATH + model_type + '_generator_latest.pth')
generator.load_state_dict(checkpoint)
generator.eval()

# create the random feature noise
if model_type == "MNIST":
    pic_number = 10
    n_feature = 100
    noise = Variable(torch.randn(pic_number, n_feature))
else:
    pic_number = 64
    n_feature = 100
    noise = torch.randn(pic_number, n_feature, 1, 1)

# generate the images
fake_images = generator(noise).detach()

# visualize the images data as showing picture on the screen
if model_type == 'MNIST':
    test_images = dm.vectors_to_images(fake_images)
    test_images = test_images.data
    dm.display_image(test_images, 0, 0)
else:
    plt.figure(figsize=(8, math.ceil(len(noise) / 8)))
    grid_img = vutils.make_grid(fake_images, normalize=True, nrow=8)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig(IMAGES_FOLDER + '/' + model_type + '_pic.png')
    plt.show()
