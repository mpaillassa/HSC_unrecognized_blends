import os
import random as rd

import numpy as np
import torch
from astropy.io import fits

IM_SIZE = 95
MARGIN = 20

hsc_mask = np.zeros([IM_SIZE, IM_SIZE])
hsc_mask[:MARGIN, :] = 1
hsc_mask[:, :MARGIN] = 1
hsc_mask[-MARGIN:, :] = 1
hsc_mask[:, -MARGIN:] = 1
hsc_mask_idx = np.where(hsc_mask)


class Blend_dataset(torch.utils.data.Dataset):
    """Class to make a custom torch dataset."""

    def __init__(self, set_dir, nb_train, preprocess_type, labels):
        self.set_dir = set_dir
        self.nb_train = nb_train
        self.preprocess_type = preprocess_type
        self.labels = labels

    def __len__(self):
        return self.nb_train

    def __getitem__(self, idx):

        im_path = os.path.join(self.set_dir, f"{idx}_stamp.fits")

        im = fits.getdata(im_path).astype(np.float32)

        if "sky_sigma" in self.preprocess_type:
            for band in range(im.shape[0]):
                im[band] -= np.mean(im[band, hsc_mask_idx[0], hsc_mask_idx[1]])
                im[band] /= np.std(im[band, hsc_mask_idx[0], hsc_mask_idx[1]])
        if "arcsinh" in self.preprocess_type:
            im = np.arcsinh(im)

        la = self.labels[idx]

        return im, la
