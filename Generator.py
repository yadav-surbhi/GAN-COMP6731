import DataManager
import torch.nn as nn


def hidden_layers(in_channel, out_channel):
    return nn.Sequential(
        nn.Linear(in_channel, out_channel),
        nn.LeakyReLU(0.2))


class Generator(nn.Module):
    def __init__(self, model_type):
        super(Generator, self).__init__()
        self.model_type = model_type
        self.n_features = 100

        if self.model_type == 'MNIST':
            n_out = 784
            self.nc = 1
            self.conv1 = hidden_layers(self.n_features, 256)
            self.conv2 = hidden_layers(256, 512)
            self.conv3 = hidden_layers(512, 1024)

            self.out = nn.Sequential(
                nn.Linear(1024, n_out),
                nn.Tanh()
            )

        else:
            # number of generator filters
            self.ngf = 64
            self.nc = 3
            self.conv1 = nn.Sequential(
                # input is n_features, going into a convolution
                nn.ConvTranspose2d(self.n_features, self.ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(self.ngf * 8),
                nn.ReLU(True))

            self.conv2 = nn.Sequential(
                nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf * 4),
                nn.ReLU(True))

            self.conv3 = nn.Sequential(
                nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf * 2),
                nn.ReLU(True))

            self.conv4 = nn.Sequential(
                nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf),
                nn.ReLU(True))

            self.out = nn.Sequential(
                nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.model_type != 'MNIST':
            x = self.conv4(x)
        x = self.out(x)
        return x

    def generate_digits(self, n=10):
        num_test_samples = n
        test_noise = DataManager.noise(num_test_samples)
        test_images = DataManager.vectors_to_images(self(test_noise))
        test_images = test_images.data
        DataManager.visualize_data(test_images, n)


def getGenerator(model_type):
    generator = Generator(model_type)
    print(generator)
    return generator
