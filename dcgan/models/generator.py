import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, in_channels, nfg, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input Size : 100 x 1 x 1
            nn.ConvTranspose2d(in_channels, nfg * 2, 7, 1, 0, bias=False),
            nn.ReLU(True),
            # Input Size : 64 x 7 x 7
            nn.ConvTranspose2d(nfg * 2, nfg * 1, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # Input Size : 32 x 14 x 14
            nn.ConvTranspose2d(nfg * 1, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # Final Output : 1 x 28 x 28
        )

    def forward(self, input):
        return self.main(input)
