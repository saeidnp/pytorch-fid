"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import Dataset

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from .inception import InceptionV3, InceptionV3Single
from .generator import GeneratorBase


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def get_activations_from_loader(loader, N, model, dims=2048, device='cpu'):
    model.eval()

    pred_arr = np.empty((N, dims))

    start_idx = 0

    for batch in tqdm(loader, desc="Feature extraction for FID"):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def compute_statistics_of_loader(loader, N, model, dims, device):
    act = get_activations_from_loader(loader=loader, N=N, model=model,
                                      dims=dims, device=device)
    m = np.mean(act, axis=0)
    s = np.cov(act, rowvar=False)
    return m, s


def compute_statistics_of_generator(generator, model, batch_size, dims, device):
    return compute_statistics_of_loader(loader=generator.loader(batch_size=batch_size),
                                        N=generator.N, model=model, dims=dims, device=device)


def compute_statistics_of_dataset(dataset, model, batch_size, dims, device,
                                  num_workers=len(os.sched_getaffinity(0))):
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)
    return compute_statistics_of_loader(loader=dataloader, N=len(dataset),
                                        model=model, dims=dims, device=device)


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=len(os.sched_getaffinity(0))):
    
    if not os.path.exists(path):
        raise RuntimeError('Invalid path: %s' % path)
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
        return m, s
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                        for file in path.glob('*.{}'.format(ext))])
        if batch_size > len(files):
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = len(files)
        dataset = ImagePathDataset(files, transforms=TF.ToTensor())

        return compute_statistics_of_dataset(dataset=dataset, model=model,
                                             batch_size=batch_size, dims=dims,
                                             device=device, num_workers=num_workers)


def compute_statistics(source, model, batch_size, dims, device,
                       num_workers=len(os.sched_getaffinity(0)),
                       cache_path=None):
    """Computes statistics required by FID based on the model's output in the
       layer specified according to "dims".

    Params:
    -- source      : Either of the following
                     - A path to a directory of images or a .npz file containing the statistics of interest,
                     - A pytorch dataset object,
                     - A generator inherited from GeneratorBase class defined in this pytorch_fid.generator.
    -- model       : Instance of inception model.
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception.
    -- device      : Device to run calculations.
    -- num_workers : Number of parallel dataloader workers.
    -- cache_path  : A path to a (non-existing) .npz file.
                     If given, stores the computed statistics at the given path.

    Returns:
    -- Statistics required by FID (mean and standard deviation of the features)
    """
    if isinstance(source, str):
        m, s = compute_statistics_of_path(path=source, model=model, batch_size=batch_size,
                                          dims=dims, device=device, num_workers=num_workers)
    elif isinstance(source, GeneratorBase):
        m, s = compute_statistics_of_generator(generator=source, model=model, batch_size=batch_size,
                                               dims=dims, device=device)
    elif isinstance(source, Dataset):
        m, s = compute_statistics_of_dataset(dataset=source, model=model, batch_size=batch_size,
                                             dims=dims, device=device, num_workers=num_workers)
    else:
        raise Exception(f"Unexpected type of \"path\" object (expected one of string or GeneratorBase, received {type(source)})")

    if cache_path is not None:
        np.savez(cache_path, mu=m, sigma=s)
    
    return m, s


def calculate_fid_given_sources(src1, src2, batch_size=64, device="cuda", dims=2048,
                                num_workers=len(os.sched_getaffinity(0)),
                                cache1=None, cache2=None):
    """Calculates the FID of two sources (path, dataset, or generator)"""

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3Single(block_idx).to(device)

    m1, s1 = compute_statistics(src1, model, batch_size,
                                dims, device, num_workers,
                                cache_path=cache1)
    m2, s2 = compute_statistics(src2, model, batch_size,
                                dims, device, num_workers,
                                cache_path=cache2)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--num-workers', type=int, default=len(os.sched_getaffinity(0)),
                        help='Number of processes to use for data loading')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('path', type=str, nargs=2,
                        help=('Paths to the generated images or '
                              'to .npz statistic files'))

    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    fid_value = calculate_fid_given_sources(args.path,
                                            args.batch_size,
                                            device,
                                            args.dims,
                                            args.num_workers)
    print('FID: ', fid_value)


if __name__ == '__main__':
    main()
