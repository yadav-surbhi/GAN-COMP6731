import json
import math

import torch.nn as nn
import DataManager as dm
import Discriminator as dism
import Generator as gen
import GAN
import torch.optim as optim
import torch
import random
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# set manual seed to a constant get a consistent output
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def main(learning_rate, model_type):
    depth = 0
    batch_size = 128

    if model_type == "MNIST":
        depth = 1
        batch_size = 100
        fixed_noise = dm.noise(int(batch_size/2))
    else:
        depth = 3
        fixed_noise = torch.randn(int(batch_size/2), 100, 1, 1)

    train_set = load_data(model_type)

    discriminator = dism.getDiscriminator(model_type)
    generator = gen.getGenerator(model_type)

    gan_model = GAN.GAN(generator, discriminator)

    # dm.visualize_data(train_set)

    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    loss_criteria = nn.BCELoss()

    model_name = model_type + "_model"
    model_path = dm.MODELS_PATH + model_name
    dm.make_dir(dm.MODELS_PATH)

    epoch = 1
    d_loss = []
    g_loss = []
    try:
        with open(dm.MODELS_PATH + model_type + '_loss.json', 'r') as filehandle:
            loss = json.load(filehandle)
            d_loss = loss[0]
            g_loss = loss[1]
    except FileNotFoundError:
        print("No saved loss data file.")

    try:
        checkpoint = torch.load(model_path)
        epoch = checkpoint['epoch']
        print("Continuing the GAN model training with epoch number :", epoch)
        if epoch > 0:
            discriminator.load_state_dict(checkpoint['discriminator_dict'])
            d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            generator.load_state_dict(checkpoint['generator_dict'])
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            loss_criteria = checkpoint['loss_criteria']
            fixed_noise = checkpoint['fixed_noise']
            discriminator.train()
            generator.train()
            epoch += 1

    except FileNotFoundError:
        print("No saved Model to be found. Let's train a new GAN model!")

    # Total number of epochs to train
    num_epochs = 200

    while True:
        print("Training the GAN net, Epoch Number : ", epoch)
        for i, (images, labels) in enumerate(train_set):

            # Train D
            real_data = getReal_data(images, model_type)
            fake_data = generator(getNoise(images, model_type)).detach()
            actual_loss, _, _ = gan_model.train_discriminator(d_optimizer, loss_criteria, real_data, fake_data)

            # Train G
            fake_data = generator(getNoise(images, model_type))
            g_error = gan_model.train_generator(g_optimizer, loss_criteria, fake_data)

            d_loss.append(actual_loss.item())
            g_loss.append(g_error.item())
            # print("loss:", actual_loss.item())

            if model_type == 'MNIST':
                if i % 200 == 0:
                    test_images = dm.vectors_to_images(generator(fixed_noise).detach())
                    test_images = test_images.data
                    dm.display_image(test_images, epoch, i)
                    plt.plot(d_loss)
                    plt.plot(g_loss)
                    save_latest_model(generator, discriminator, model_type, epoch)
            else:
                if i % 50 == 0:
                    fake_images = generator(fixed_noise).detach()

                    plt.figure(figsize=(8, math.ceil(len(fixed_noise) / 8)))
                    grid_img = vutils.make_grid(fake_images, normalize=True, nrow=8)
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.imshow(grid_img.permute(1, 2, 0))
                    plt.show()

                    dm.make_dir(dm.IMAGES_FOLDER + "/CIFAR")
                    vutils.save_image(fake_images, dm.IMAGES_FOLDER + '/CIFAR/epoch_%03d_batch_%03d.png' % (epoch, i),
                                      normalize=True)
                    save_latest_model(generator, discriminator, model_type, epoch)

        loss = [d_loss, g_loss]
        with open(dm.MODELS_PATH + model_type + '_loss.json', 'w') as filehandle:
            json.dump(loss, filehandle)
        torch.save({
            'epoch': epoch,
            'discriminator_dict': discriminator.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'generator_dict': generator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'loss_criteria': loss_criteria,
            'fixed_noise': fixed_noise
        }, model_path)

        print("Saved Train Epoch:", epoch)
        epoch += 1


def load_data(model_type, ):
    if model_type == 'MNIST':
        train_set = dm.load_MNIST(100)
    else:
        train_set = dm.load_CIFAR10(128)

    return train_set


def getReal_data(images, model_type):
    if model_type == 'MNIST':
        return images.view(images.size(0), 28 * 28)
    else:
        return images


def getNoise(images, model_type):
    batch_size = images.size(0)
    if model_type == 'MNIST':
        return dm.noise(batch_size)
    else:
        return torch.randn(batch_size, 100, 1, 1)


def save_latest_model(generator, discriminator, model_type, epoch):
    # torch.save(generator.state_dict(),
    #          dm.MODELS_PATH + '/' + model_type + '_generator_epoch_%03d.pth' % epoch)
    torch.save(generator.state_dict(),
               dm.MODELS_PATH + '/' + model_type + '_generator_latest.pth')


print('Specify the Dataset:MNIST or CIFAR')
x =input()
while x not in ["MNIST","CIFAR"]:
    print('Specify the Dataset:MNIST or CIFAR')
    x =input()
# main(0.0002, 'MNIST')
main(0.0002,x)

#https://arxiv.org/abs/1511.06434
#https://github.com/Ksuryateja/DCGAN-CIFAR10-pytorch
#https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
