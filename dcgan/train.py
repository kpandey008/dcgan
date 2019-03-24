import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import torchvision.utils as vutils

from utils.visualization import plot_training_images, plot_images
from utils.config import ConfigLoader
from loaders import mnist_loader
from loaders.mnist import MNISTDataset
from models.generator import Generator
from models.discriminator import Discriminator

config = ConfigLoader('gan.conf')

# plot_training_images(dataset, 64, (28, 28), shuffle=True)


# define the weight initialization strategy
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

# Create the Generator
ngpu = int(config.get_param_value('train', 'num_gpu'))
z = int(config.get_param_value('generator', 'z'))
nfg = int(config.get_param_value('generator', 'nfg'))
nfd = int(config.get_param_value('discriminator', 'nfd'))
nc = int(config.get_param_value('train', 'num_channels'))
learning_rate = float(config.get_param_value('train', 'learning_rate'))
beta1 = float(config.get_param_value('train', 'beta1'))
batch_size = int(config.get_param_value('train', 'batch_size'))

num_epochs = int(config.get_param_value('train', 'num_epochs'))

device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

# move the models to the device
netG = Generator(ngpu, z, nfg, nc).to(device)
netD = Discriminator(ngpu, nfd, nc).to(device)

# Handle multi gpu
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))

netG.apply(weight_init)
netD.apply(weight_init)

print(netG)
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, z, 1, 1, device=device)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

D_loss = []
G_loss = []
image_list = []
iters = 0
# TODO : Remove the training epochs and replace with training steps

# Training Loop
for epoch_idx in range(num_epochs):
    for i, data in enumerate(mnist_loader, 0):
        ##############################
        ###  Disriminator training ###
        ##############################

        # reset the gradients
        netD.zero_grad()

        # transfer the images to the specified device (required for
        # training on a gpu)
        real_images = data[0].to(device)
        labels = torch.full((batch_size,), real_label, device=device)

        # Forward pass the real batch through discriminator
        output = netD(real_images).view(-1)
        prob_D = output.mean().item()
        errD_real = criterion(output, labels)
        errD_real_mean = output.mean().item()

        # generate fake images
        noise_batch = torch.randn(batch_size, z, 1, 1, device=device)
        fake_images = netG(noise_batch).to(device)
        labels = torch.full((batch_size,), fake_label, device=device)

        # Forward pass the fake batch through discriminator
        output = netD(fake_images.detach()).view(-1)
        errD_fake = criterion(output, labels)
        errD_fake_mean = output.mean().item()

        errD = errD_real + errD_fake
        errD.backward()

        # Make a gradient step
        optimizerD.step()

        ##############################
        ####  Generator training  ####
        ##############################

        netG.zero_grad()

        # create another batch of fakes for the generator
        noise_batch = torch.randn(batch_size, z, 1, 1, device=device)  
        fake_images = netG(noise_batch).to(device)
        labels = torch.full((batch_size,), real_label, device=device)

        # Forward pass the fake batch through generator
        output = netD(fake_images).view(-1)
        prob_G = output.mean().item()
        errG = criterion(output, labels)
        errG_mean = output.mean().item()

        errG.backward()
        optimizerG.step()

        # Record the training stats
        if i % 50 == 0:
            print(f"[{epoch_idx}/{num_epochs}][{i}/{len(mnist_loader)}]\tLoss_D: {errD.item()}\tLoss_G: {errG.item()}\tProb D: {prob_D}\tProb G: {prob_G}")

        # Save the loss profiles for plotting later
        D_loss.append(errD.item())
        G_loss.append(errG.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch_idx == num_epochs - 1) and (i == len(mnist_loader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            image_list.append(fake)

        iters += 1
plot_images(image_list[-1])
