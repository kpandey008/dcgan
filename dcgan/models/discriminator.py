import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ngpu, in_channels, nfd, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input Size: 3 x 64 x 64
            nn.Conv2d(nc, nfd, 4, 2, 1, bias=False),
            nn.LeakyReLU(True),
            # Input Size: 32 x 32 x 32
            nn.Conv2d(nfd, nfd * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(True),
            # Input Size: 64 x 16 x 16
            nn.Conv2d(nfd * 2, nfd * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(True),
            # Input Size: 128 x 8 x 8
            nn.Conv2d(nfd * 4, nfd * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(True),
            # Input Size: 256 x 4 x 4
            nn.Conv2d(nfd * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # Output Size: 1 x 1 x 1
        )

    def forward(self, input):
        return self.main(input)
