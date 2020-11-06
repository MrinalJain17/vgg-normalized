"""
This code is used to normalize the weights of the VGG network as stated in the
paper "A Neural Algorithm of Artistic Style". The code is adapted from the following
keras implementation: https://github.com/corleypc/vgg-normalize
"""

from argparse import ArgumentParser
from multiprocessing import cpu_count

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MeanActivations(nn.Module):
    def __init__(self):
        super(MeanActivations, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        mean_activations = []

        x = self.vgg[0](x)
        for layer in self.vgg[1:]:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                mean_activations.append(x.mean(dim=(0, 2, 3)))

        return mean_activations


def val_dataloader(valdir, batch_size):
    return DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=cpu_count(),
        pin_memory=True,
        drop_last=True,
    )


def get_mean_activations(valdir, batch_size=64):
    val_loader = val_dataloader(valdir=valdir, batch_size=batch_size)
    model = MeanActivations().to(device)
    model.eval()

    with torch.no_grad():
        accumulated_means = None
        for (images, target) in tqdm(val_loader, unit="batch"):
            images = images.to(device)
            target = target.to(device)

            batch_means = model(images)
            if accumulated_means is None:
                accumulated_means = batch_means
            else:
                for acc, m in zip(accumulated_means, batch_means):
                    acc += m

        for acc in accumulated_means:
            acc /= len(val_loader)

    return accumulated_means


def normalize_vgg_weights(layer_means, save_path=None):
    model = models.vgg19(pretrained=True).to(device)

    with torch.no_grad():
        previous_layer_mean = None
        iterator = iter(layer_means)
        for layer in model.features[1:]:
            if isinstance(layer, nn.Conv2d):
                current_mean = next(iterator)
                weights, bias = (
                    layer.weight.data,
                    layer.bias.data,
                )  # Shape of weights: (out_channels, in_channels, dim_1, dim_2)
                if previous_layer_mean is not None:
                    weights *= previous_layer_mean[None, :, None, None]

                weights /= current_mean[:, None, None, None]
                bias /= current_mean

                previous_layer_mean = current_mean

    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--validation_path", type=str, default="./ILSVRC2012_img_val")
    parser.add_argument("--save_path", type=str, default="./vgg19_normalized.pth")
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()

    activation_means = get_mean_activations(args.validation_path, args.batch_size)
    normalize_vgg_weights(activation_means, args.save_path)
