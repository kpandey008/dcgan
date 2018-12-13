import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ngpu, nfd, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input Size: 1 x 28 x 28
            nn.Conv2d(nc, nfd, 4, 2, 1, bias=False),
            nn.LeakyReLU(True),
            # Input Size: 32 x 14 x 14
            nn.Conv2d(nfd, nfd * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(True),
            # Input Size: 64 x 7 x 7
            nn.Conv2d(nfd * 2, 1, 7, 1, 0, bias=False),
            nn.Sigmoid(),
            # Output Size : 1 x 1 x 1
        )

    def forward(self, input):
        return self.main(input)
