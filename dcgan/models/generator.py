import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, in_channels, nfg, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, nfg * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nfg * 8, nfg * 4, 4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nfg * 4, nfg * 2, 4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nfg * 2, nfg, 4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nfg, nc, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
