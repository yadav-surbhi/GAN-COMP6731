import torch


class GAN:

    def __init__(self, generator, discriminator):
        self.discriminator = discriminator
        discriminator.apply(weights_init)
        self.generator = generator
        generator.apply(weights_init)

    def train_generator(self, optimizer, loss, fake_data):
        batch_size = fake_data.size(0)

        # Reset gradients
        optimizer.zero_grad()

        # Sample noise and generate fake data
        prediction = self.discriminator(fake_data)

        label = torch.full((batch_size,), 1, dtype=torch.float)

        error = loss(prediction, label)

        error.backward()

        optimizer.step()

        # Return error
        return error

    def train_discriminator(self, optimizer, loss, real_data, fake_data):
        # Reset gradients
        optimizer.zero_grad()

        # Train on Real Data
        batch_size = real_data.size(0)
        prediction_real = self.discriminator(real_data)

        label = torch.full((batch_size,), 1, dtype=torch.float)

        error_real = loss(prediction_real, label)
        accuracy = (prediction_real - label).float().mean()
        error_real.backward()

        # Train on Fake Data
        prediction_fake = self.discriminator(fake_data)

        label.fill_(0)
        error_fake = loss(prediction_fake, label)
        error_fake.backward()

        optimizer.step()

        # Return error and predictions for real and fake inputs
        return error_real + error_fake, prediction_real, prediction_fake


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
