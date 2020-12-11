import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from baseline import BasicBaseline, FederatedBaseline
from utils.basics import load_model, imshow, save_model, imsave
from utils.models import AttackGenerator


def main(verbose=True):
    print_every = 10

    discriminator = BasicBaseline().model
    load_model(discriminator, "basic")

    generator = AttackGenerator(input_dim=10, output_dim=1)
    generator.train()

    tool = FederatedBaseline(num_clients=10)
    tool.load_data()
    trainloader = tool._make_client_trainloaders()[0]

    batch_size = 32
    z_dim = 10

    num_epochs = 30
    lr = 1e-3
    optimizer = optim.Adam(generator.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    fixed_noise = torch.rand((batch_size, z_dim))

    train_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0], data[1]
            z = torch.rand((labels.size()[0], z_dim))

            optimizer.zero_grad()

            gen_fake = generator(z)
            dis_fake = discriminator(gen_fake)
            loss = criterion(dis_fake, labels)
            loss.backward()
            optimizer.step()

            if verbose:
                running_loss += loss.item()
                if i % print_every == 0 and i != 0:  
                    print(f"[epoch: {epoch}, datapoint: {i}] \t loss: {round(running_loss / print_every, 3)}")
                    running_loss = 0.0
                    test(generator, fixed_noise, epoch)
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(trainloader)) #this is buggy

    return generator, train_losses


    
def test(model, z, epoch):
    model.eval()
    pred = model(z)
    model.train()
    
    num_samples = 32
    image_frame_dim = int(np.floor(np.sqrt(num_samples)))
    samples = pred.data.numpy().transpose(0, 2, 3, 1)
    # samples = (samples + 1) / 2
    samples *= 255
    imsave(samples[:image_frame_dim * image_frame_dim, :, :, :].astype(np.uint8), [image_frame_dim, image_frame_dim], f"figures/gan_fig_epoch{epoch}.png")


if __name__ == "__main__":
    generator, losses = main()
    save_model(generator, "gan_15epochs")