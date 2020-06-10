
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image
import numpy as np

from .model import Generator, Discriminator

from torchvision.models.vgg import vgg19

class VGGLoss(nn.Module):
    def __init__(self, loss_type, device):
        super(VGGLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        if loss_type == 'vgg22':
            vgg_net = nn.Sequential(*list(vgg.features[:9]))
        elif loss_type == 'vgg54':
            vgg_net = nn.Sequential(*list(vgg.features[:36]))
        
        for param in vgg_net.parameters():
            param.requires_grad = False

        vgg_net.to(device)

        self.vgg_net = vgg_net.eval()
        self.mse_loss = nn.MSELoss()

        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406], requires_grad=False, device=device))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225], requires_grad=False, device=device))

    def forward(self, real_img, fake_img):
        real_img = real_img.sub(self.vgg_mean[:, None, None]).div(self.vgg_std[:, None, None])
        fake_img = fake_img.sub(self.vgg_mean[:, None, None]).div(self.vgg_std[:, None, None])
        feature_real = self.vgg_net(real_img)
        feature_fake = self.vgg_net(fake_img)
        return self.mse_loss(feature_real, feature_fake)

def train(
    epochs,
    dataset,
    G,
    optimizer_G,
    scheduler_G,
    D,
    optimizer_D,
    scheduler_D,
    validity_criterion,
    content_criterion,
    validity_gamma,
    device,
    verbose_interval,
    save_interval
):

    patch = (1, 16, 16)
    losses = {
        'g' : [],
        'd' : []
    }
    
    for epoch in range(epochs):
        for index, (target, pair) in enumerate(dataset, 1):
            # label smoothing
            gt   = torch.from_numpy((1.0 - 0.7) * np.random.randn(target.size(0), *patch) + 0.7)
            gt   = gt.type(torch.FloatTensor).to(device)
            fake = torch.from_numpy((0.3 - 0.0) * np.random.randn(target.size(0), *patch) + 0.0)
            fake = fake.type(torch.FloatTensor).to(device)

            # target image
            target = target.type(torch.FloatTensor).to(device)
            # pair image
            pair = pair.type(torch.FloatTensor).to(device)


            # generate image
            gen_image = G(pair)


            '''
            Train Discriminator
            '''

            # real
            pred_real = D(target)
            d_real_loss = validity_criterion(pred_real, gt)

            # fake
            pred_fake = D(gen_image.detach())
            d_fake_loss = validity_criterion(pred_fake, fake)

            d_loss = (d_real_loss + d_fake_loss) / 2

            # optimize
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()


            '''
            Train Generator
            '''


            # train to fool D
            pred_fake = D(gen_image)
            g_validity_loss = validity_criterion(pred_fake, gt)
            g_content_loss = content_criterion((gen_image + 1) / 2, target)
            # g_mse_loss = validity_criterion(gen_image, target)
            g_loss = 1.e-3 * g_validity_loss + g_content_loss

            # optimize
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            scheduler_G.step()
            scheduler_D.step()



            losses['g'].append(g_loss.item())
            losses['d'].append(d_loss.item())

            batches_done = epoch * len(dataset) + index

            if batches_done % verbose_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, epochs, index, len(dataset), d_loss.item(), g_loss.item())
                )
            

            if batches_done % save_interval == 0 or batches_done == 1:
                num_images = 32
                images = grid(pair, gen_image, target, num_images)
                save_image(images, "SRGAN/result/%d.png" % batches_done, nrow=4*3)

        torch.save(G.state_dict(), 'SRGAN/models/generator_{}.pt'.format(epoch))
        

    return losses

def grid(pair, gen_image, target, num_images):
    image_shape = target.size()[1:]

    pair = F.pad(pair, pad=(96, 96, 96, 96), mode='constant', value=-1.)

    pair_tuple   = torch.unbind(pair)
    gen_tuple    = torch.unbind(gen_image)
    target_tuple = torch.unbind(target)

    images = []
    for pair, gen_image, target in zip(pair_tuple, gen_tuple, target_tuple):
        images = images + [pair.view(1, *image_shape), gen_image.view(1, *image_shape), target.view(1, *image_shape)]

        if len(images) >= num_images*3:
            break
    return torch.cat(images)


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def plot_loss(losses):
    g_losses = losses['g']
    d_losses = losses['d']

    plt.figure(figsize=(12, 8))

    plt.plot(g_losses)
    plt.plot(d_losses)

    plt.legend(['generator', 'discriminator'])
    plt.xlabel('n_iter')
    plt.ylabel('loss')

    plt.savefig('/usr/src/SRGAN/curve.png')

def main(
    dataset_class,
    to_loader
):

    image_size = 256
    n_images = 24000
    epochs = 150
    lr = 0.0001
    betas = (0.9, 0.999)
    validity_gamma = 1.e-3

    batch_size = 24

    import torchvision.transforms as transforms

    pair_transform = transforms.Resize((image_size // 4, image_size // 4))
    dataset = dataset_class(pair_transform=pair_transform, image_size=256, n_images=n_images)
    dataset = to_loader(dataset, batch_size)

    G = Generator(n_blocks=16)
    D = Discriminator()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=betas)

    def lr_lambda(iter):
        if iter < 1e5:
            return 1.e-4
        else:
            return 1.e-5

    scheduler_G = LambdaLR(optimizer_G, lr_lambda=lr_lambda)
    scheduler_D = LambdaLR(optimizer_D, lr_lambda=lr_lambda)

    validity_criterion = nn.MSELoss()
    content_criterion = VGGLoss(loss_type='vgg54', device=device)

    losses = train(
        epochs=epochs,
        dataset=dataset,
        G=G,
        optimizer_G=optimizer_G,
        scheduler_G=scheduler_G,
        D=D,
        optimizer_D=optimizer_D,
        scheduler_D=scheduler_D,
        validity_criterion=validity_criterion,
        content_criterion=content_criterion,
        validity_gamma=validity_gamma,
        device=device,
        verbose_interval=100,
        save_interval=1000
    )

    plot_loss(losses)

