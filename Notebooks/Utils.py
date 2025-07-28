import torch
import torch.onnx
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F

DATA_PATH = '../data'


def get_device():
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    print(f'Using device: {device}')
    return device


def get_loader(dataset, x_dim, batch_size, normalize=True):
    if normalize:
        trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize((x_dim, x_dim)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize((x_dim, x_dim)),
            torchvision.transforms.ToTensor()
        ])
    dataset = dataset(
        root=DATA_PATH,
        download=True,
        transform=trans
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print('Data loaded')
    return dataloader


def save_model(model, dummy_noise, name, dummy_labels=None):
    model.eval()
    if dummy_labels is None:
        torch.onnx.export(
            model,
            dummy_noise,  # Latent space input (a dummy one, so that ONNX knows the shapes)
            f"../Models/{name}.onnx",
            input_names=["latent_vector"],
            output_names=["generated_image"],
            dynamic_axes={"latent_vector": {0: "batch_size"}, "generated_image": {0: "batch_size"}},
            opset_version=11
        )
    else:
        torch.onnx.export(
            model,
            (dummy_noise, dummy_labels),  # Latent space input (a dummy one, so that ONNX knows the shapes)
            f"../Models/{name}.onnx",
            input_names=["latent_vector", "class_label"],
            output_names=["generated_image"],
            dynamic_axes={
                "latent_vector": {0: "batch_size"},
                "class_label": {0: "batch_size"},
                "generated_image": {0: "batch_size"}
            },
            opset_version=11
        )


def get_n_params(model):
    return sum(p.numel() for p in model.parameters())


def plot_real(dataloader, cnt=32):
    real_batch = next(iter(dataloader))[0]  # shape: (B, C, H, W)
    images = real_batch.permute(0, 2, 3, 1).cpu().numpy()  # shape: (B, H, W, C)

    if images.shape[-1] == 1:
        images = images.squeeze(-1)  # shape: (B, H, W) for grayscale

    vmin = -1 if images.min() < 0 else 0
    vmax = 1 if images.max() <= 1 else 255

    plot_images(images, cnt=cnt, vmin=vmin, vmax=vmax, columns=16, height=1.2, width=1.2)


def plot_images(images, cnt=32, vmin=0, vmax=255, columns=6, height=2., width=2.):
    images = (images - vmin) / (vmax - vmin)
    rows = -(-len(images) // columns)
    plt.figure(figsize=(columns * width, rows * height))
    cmap = 'gray' if (images.ndim == 3 or images.shape[-1] == 1) else None
    for i, image in enumerate(images[:cnt]):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image, vmin=0, vmax=1, cmap=cmap)
        plt.axis(False)
    plt.show()


# VAE

def VAE_train(netE, netD, optimizer, criterion, dataloader, epoch_num, device):
    for epoch in range(1, epoch_num + 1):
        for i, data in enumerate(dataloader, 0):
            data = data[0].to(device)

            z, mu, log_var = netE(data)
            x_hat = netD(z)

            loss = criterion(x_hat, data, mu, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Output training stats
        print('[%d/%d][%d/%d]\tLoss: %.4f' % (epoch, epoch_num, i, len(dataloader), loss.item()))

        # Every tenth epoch, plot fake images
        if epoch % 25 == 0:
            fake_batch = netD.generate(32).detach().cpu().reshape(-1, 28, 28, 1)
            plot_images(fake_batch, vmin=0, vmax=1, columns=16, height=1.2, width=1.2)


def vae_loss(reconstructed, original, mu, log_var):
    recon_loss = F.binary_cross_entropy(reconstructed, original, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_div


# GAN

def save_checkpoint(c_epoch, netD, optimizerD, netG, optimizerG):
    latest_checkpoint_path = f'Checkpoints/checkpoint_epoch_{c_epoch}.pt'
    checkpoint = {'netD_state_dict': netD.state_dict(),
                  'netG_state_dict': netG.state_dict(),
                  'optimizerD_state_dict': optimizerD.state_dict(),
                  'optimizerG_state_dict': optimizerG.state_dict()}
    torch.save(checkpoint, latest_checkpoint_path)
    return latest_checkpoint_path


def load_checkpoint(netD, optimizerD, netG, optimizerG, latest_checkpoint_path, device, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = latest_checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    netD.load_state_dict(checkpoint['netD_state_dict'])
    netG.load_state_dict(checkpoint['netG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    print(f"Checkpoint loaded from {checkpoint_path}.")


def GAN_train(netD, netG, optimizerD, optimizerG, criterion, dataloader,
              epoch_num, x_dim, z_dim, viz_noise, device, fake_label=0, real_label=1,
              checkpoint_num=None, collapse_ratio=500):
    latest_checkpoint_path = None

    if checkpoint_num is not None:
        load_checkpoint(netD, optimizerD, netG, optimizerG, latest_checkpoint_path, device,
                        'Checkpoints/checkpoint_epoch_{checkpoint_num}.pt')

    for epoch in range(1, epoch_num + 1):
        for i, data in enumerate(dataloader, 0):
            # (1) Update the discriminator with real data
            optimizerD.zero_grad()
            # Format batch
            real = data[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # (2) Update the discriminator with fake data
            # Generate batch of latent vectors
            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach())
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # (3) Update the generator with fake data
            optimizerG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i != 0 and i % 150 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, epoch_num, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Check for mode collapse:
        if epoch >= 3 and errG / errD.item() > collapse_ratio:
            print(f"Potential mode collapse detected: Loss_D = {errD.item():.4f} and Loss_G = {errG.item():.4f}.")
            load_checkpoint(netD, optimizerD, netG, optimizerG, latest_checkpoint_path, device)
            continue

        # Every fifth epoch, plot fake images
        if epoch % 5 == 0:
            with torch.no_grad():
                fake = netG(viz_noise).detach().cpu()
            latest_checkpoint_path = save_checkpoint(epoch, latest_checkpoint_path, netD, optimizerD, netG, optimizerG)
            fake_batch = fake[:32].reshape(-1, x_dim, x_dim, 1)
            plot_images(fake_batch, vmin=-1, vmax=1, columns=16, height=1.2, width=1.2)


def gan_plot_fake(generator, noise, x_dim, cnt=32):
    with torch.no_grad():
        fake = generator(noise).detach().cpu()

    fake_batch = fake.permute(0, 2, 3, 1)
    plot_images(fake_batch, cnt=cnt, vmin=-1, vmax=1, columns=16, height=1.2, width=1.2)


def cGAN_train(netD, netG, optimizerD, optimizerG, criterion, dataloader,
               epoch_num, x_dim, z_dim, viz_noise, viz_labels, device, fake_label=0, real_label=1,
               checkpoint_num=None, collapse_ratio=500):
    latest_checkpoint_path = None

    if checkpoint_num is not None:
        load_checkpoint(netD, optimizerD, netG, optimizerG, latest_checkpoint_path, device,
                        f'Checkpoints/checkpoint_epoch_{checkpoint_num}.pt')

    for epoch in range(1, epoch_num + 1):
        for i, data in enumerate(dataloader, 0):
            # (1) Update the discriminator with real data
            optimizerD.zero_grad()
            # Format batch
            real, class_labels = data[0].to(device), data[1].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real, class_labels)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # (2) Update the discriminator with fake data
            # Generate batch of latent vectors
            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise, class_labels)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach(), class_labels)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # (3) Update the generator with fake data
            optimizerG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake, class_labels)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i != 0 and i % 150 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, epoch_num, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Check for mode collapse:
        if epoch >= 3 and errG / errD.item() > collapse_ratio:
            print(f"Potential mode collapse detected: Loss_D = {errD.item():.4f} and Loss_G = {errG.item():.4f}.")
            load_checkpoint(netD, optimizerD, netG, optimizerG, latest_checkpoint_path, device)
            continue

        # Every fifth epoch, plot fake images
        if epoch % 5 == 0:
            with torch.no_grad():
                fake = netG(viz_noise, viz_labels).detach().cpu()
            latest_checkpoint_path = save_checkpoint(epoch, latest_checkpoint_path, netD, optimizerD, netG, optimizerG)
            fake_batch = fake[:32].permute(0, 2, 3, 1)
            plot_images(fake_batch, vmin=-1, vmax=1, columns=16, height=1.2, width=1.2)


def cgan_plot_fake(generator, noise, labels, x_dim, cnt=32):
    with torch.no_grad():
        fake = generator(noise, labels).detach().cpu()

    fake_batch = fake.permute(0, 2, 3, 1)
    plot_images(fake_batch, cnt=cnt, vmin=-1, vmax=1, columns=16, height=1.2, width=1.2)
