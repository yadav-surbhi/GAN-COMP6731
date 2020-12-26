import torch.nn as nn
import DataManager as dm
import Discriminator as dism
import Generator as gen
import GAN
import torch.optim as optim
import torch


def main(learning_rate, d):
    train_set = dm.load_MNIST(d)
    discriminator = dism.getDiscriminator()
    generator = gen.getGenerator()

    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()

    model = GAN.GAN(generator, discriminator)

    # dm.visualize_data(train_set)

    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
    loss_criteria = nn.BCELoss()

    epoch = 1
    try:
        checkpoint = torch.load(dm.MODELS+"_"+str(d))
        epoch = checkpoint['epoch']
        print("Continuing the GAN model training with epoch number :", epoch)
        if epoch > 0:
            discriminator.load_state_dict(checkpoint['discriminator_dict'])
            d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            generator.load_state_dict(checkpoint['generator_dict'])
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            loss_criteria = checkpoint['loss_criteria']
            discriminator.train()
            generator.train()
            epoch += 1
    except FileNotFoundError:
        print("No saved Model to be found. Let's train a new GAN model!")

    # Total number of epochs to train
    num_epochs = 200

    while True:
        print("Training the GAN net, Epoch Number : ", epoch)
        d_loss = []
        g_loss = []
        for i, (images, labels) in enumerate(train_set):

            # Train D
            real_data = images.view(images.size(0), 28 * 28)
            if torch.cuda.is_available():
                real_data = real_data.cuda()
            fake_data = generator(dm.noise(images.size(0))).detach()
            actual_loss, _, _ = model.train_discriminator(d_optimizer, loss_criteria, real_data, fake_data)

            # Train G
            fake_data = generator(dm.noise(images.size(0)))
            g_error = model.train_generator(g_optimizer, loss_criteria, fake_data)

            d_loss.append(actual_loss.item())
            g_loss.append(g_error.item())
            # print("loss:", actual_loss.item())

            if i % 200 == 0:
                test_images = dm.vectors_to_images(generator(dm.noise(10)))
                test_images = test_images.data
                dm.display_image(test_images, epoch, i, d)

        torch.save({
            'epoch': epoch,
            'discriminator_dict': discriminator.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'generator_dict': generator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'loss_criteria': loss_criteria,
        }, dm.MODELS+"_"+str(d))

        print("Saved Train Epoch:", epoch)
        epoch += 1


main(0.0002, 3)
