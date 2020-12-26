import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, model_type):
        super(Discriminator, self).__init__()
        self.model = model_type

        if self.model == 'MNIST':
            self.nc = 1
            self.conv1 = self.hidden_layers(28 * 28, 1024)
            self.conv2 = self.hidden_layers(1024, 512)
            self.conv3 = self.hidden_layers(512, 256)

            self.conv4 = nn.Sequential(
                nn.Linear(256, 1),
                nn.Sigmoid())
        else:
            # number of discriminator filters
            self.ndf = 64
            self.nc = 3

            self.conv1 = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True))

            self.conv2 = nn.Sequential(
                nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 2),
                nn.LeakyReLU(0.2, inplace=True))

            self.conv3 = nn.Sequential(
                nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 4),
                nn.LeakyReLU(0.2, inplace=True))

            self.conv4 = nn.Sequential(
                nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 8),
                nn.LeakyReLU(0.2, inplace=True))

            self.out = nn.Sequential(
                nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid())

    def hidden_layers(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if self.model != 'MNIST':
            x = self.out(x)

        return x.view(-1, 1).squeeze(1)


def getDiscriminator(model_type):
    discriminator = Discriminator(model_type)
    print(discriminator)
    return discriminator
