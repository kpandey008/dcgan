import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, in_channels, nfg, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input Size : 100 x 1 x 1
            nn.ConvTranspose2d(in_channels, nfg * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            # Input Size : 256 x 4 x 4
            nn.ConvTranspose2d(nfg * 8, nfg * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # Input Size : 128 x 8 x 8
            nn.ConvTranspose2d(nfg * 4, nfg * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # Input Size : 64 x 16 x 16
            nn.ConvTranspose2d(nfg * 2, nfg, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # Input Size : 32 x 32 x 32
            nn.ConvTranspose2d(nfg, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final Output: 3 x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
