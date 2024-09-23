import os
import random as rd
import sys
from io import StringIO

import data
import matplotlib.pyplot as plt
import model
import numpy as np
import torch
import tqdm
from astropy.io import fits

IM_SIZE = 95
MARGIN = 20

hsc_mask = np.zeros([IM_SIZE, IM_SIZE])
hsc_mask[:MARGIN, :] = 1
hsc_mask[:, :MARGIN] = 1
hsc_mask[-MARGIN:, :] = 1
hsc_mask[:, -MARGIN:] = 1
hsc_mask_idx = np.where(hsc_mask)


class Capturing(list):
    """Class to create a context manager that captures stdout"""

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def get_labels(set_dirs, nb_samples):
    """Gets labels for the requested sets.
    Either read them from disk and compute them and write them to disk

    Args:
        - set_dirs (list): list of sets to get labels for.
        - nb_samples (list): list of number of samples of each set.
    Returns:
        - set_labels (list): list of labels of each set.
    """

    set_labels = []
    for i, set_dir in enumerate(set_dirs):
        set_dir_labels_path = f"{set_dir}_labels.npy"
        if not os.path.isfile(set_dir_labels_path):
            labels = np.zeros([nb_samples[i]])
            for k in tqdm.tqdm(range(nb_samples[i]), desc=f"Set {set_dir} labels"):
                sample_path = os.path.join(set_dir, f"{k}_stamp.fits")
                sample_hd = fits.getheader(sample_path, 1)

                try:
                    _ = sample_hd["IDENT_1_1"]
                    labels[k] = 1
                except KeyError:
                    labels[k] = 0
            np.save(set_dir_labels_path, labels)
        else:
            labels = np.load(set_dir_labels_path)

        set_labels.append(labels)

    return set_labels


def setup_model_dir(model, optim, out_dir, params, preprocess_type):
    """Setups the model output directory.
    This involves creating a 4-digit unique number id for the model to have a model dedicated directory and make comparisons between models later.

    Args:
        - model (torch.nn.Module): blend identifier model.
        - optim (torch.optim.Optimizer): model optimizer.
        - params (dict): training params to dump for record.
    Returns:
        - model_dir (string): model directory used for saving and training information.
    """

    # pick an id for the model
    model_id = rd.randint(1000, 9999)
    model_dir = os.path.join(out_dir, f"model{model_id:04d}")
    while os.path.isdir(model_dir):
        model_id = rd.randint(1000, 9999)
        model_dir = os.path.join(out_dir, f"model{model_id:04d}")

    # make the directory
    os.mkdir(model_dir)

    # dump information
    with open(os.path.join(model_dir, "model.cfg"), "w") as cfgfile:
        for key, value in params.items():
            cfgfile.write(f"{key}: {value}\n")
        cfgfile.write(f"preprocess type: {preprocess_type}\n")
        with Capturing() as model_summary:
            print(model)
        cfgfile.write("Model summary:\n")
        for summary_element in model_summary:
            cfgfile.write(f"{summary_element}\n")
        with Capturing() as opt_summary:
            print(optim)
        cfgfile.write("Optimizer summary:\n")
        for opt_element in opt_summary:
            cfgfile.write(f"{opt_element}\n")

    return model_dir


def make_inference(
    blend_model,
    device,
    set_dir,
    idx_beg,
    idx_end,
    labels,
    preprocess_type,
    model_dir,
    name,
):

    preds = np.zeros([idx_end - idx_beg])

    for k in tqdm.tqdm(
        range(idx_beg, idx_end), desc=f"{set_dir} inference {idx_beg} to {idx_end}"
    ):

        im_path = os.path.join(set_dir, f"{k}_stamp.fits")
        im = fits.getdata(im_path).astype(np.float32)

        if "sky_sigma" in preprocess_type:
            for band in range(im.shape[0]):
                im[band] -= np.mean(im[band, hsc_mask_idx[0], hsc_mask_idx[1]])
                im[band] /= np.std(im[band, hsc_mask_idx[0], hsc_mask_idx[1]])
        if "arcsinh" in preprocess_type:
            im = np.arcsinh(im)

        im = np.expand_dims(im, 0)
        pred = blend_model(torch.from_numpy(im).to(device))
        preds[k - idx_beg] = pred.cpu().detach().numpy()[0, 1]

    save_path = os.path.join(model_dir, f"{name}.npy")
    np.save(save_path, preds)

    acc = np.mean((preds > 0.5) == labels[idx_beg:idx_end])
    with open(os.path.join(model_dir, "acc.txt"), "a") as facc:
        facc.write(f"preprocessing {preprocess_type} {name} accuracy: {acc:.2f}\n")


def main():

    # data and directories parameters
    data_dir = "/virtualroot/work01/maxime"
    set_dir_ms8 = os.path.join(data_dir, f"set_with_hst_ms8_release_run")
    set_dir_ms4 = os.path.join(data_dir, f"set_with_hst_ms4_release_run")
    training_set = "_ms8"
    out_dir = os.path.join(data_dir, "torch")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # training parameters
    params = {"batch_size": 1024, "nb_train": 150000, "nb_test": 50000, "nb_epochs": 40}
    set_dirs, nb_samples = [set_dir_ms8, set_dir_ms4], [
        params["nb_train"] + params["nb_test"],
        params["nb_train"] + params["nb_test"],
    ]
    labels_ms8, labels_ms4 = get_labels(set_dirs, nb_samples)

    # preprocess_type
    for preprocess_type in ["none", "sky_sigma", "arcsinh", "sky_sigma_arcsinh"]:

        # data pipeline
        if training_set == "_ms8":
            dataset = data.Blend_dataset(
                set_dir_ms8, params["nb_train"], preprocess_type, labels_ms8
            )
        elif training_set == "_ms4":
            dataset = data.Blend_dataset(
                set_dir_ms4, params["nb_train"], preprocess_type, labels_ms4
            )
        data_loader = torch.utils.data.DataLoader(
            dataset, shuffle=True, batch_size=params["batch_size"]
        )

        # model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blend_model = model.BlendIdentifier(drop_rate=0.25).to(device)

        # optimizer
        optim = torch.optim.AdamW(
            blend_model.parameters(), lr=0.001, weight_decay=0.001
        )

        # setup model directory and information
        model_dir = setup_model_dir(
            blend_model, optim, out_dir, params, preprocess_type
        )
        save_dir = os.path.join(model_dir, "save")
        os.mkdir(save_dir)

        # training loop
        torch_loss = torch.nn.CrossEntropyLoss()
        loss_record = []
        for epoch in tqdm.tqdm(range(params["nb_epochs"]), desc="Epochs"):
            blend_model.train()

            for step_idx, (images, labels) in tqdm.tqdm(
                enumerate(data_loader), desc="Training steps"
            ):
                images, labels = (images.to(device), labels.to(device))
                preds = blend_model(images)
                loss = torch_loss(preds, labels.type(torch.long))

                optim.zero_grad()
                loss.backward()
                optim.step()

                loss_record.append(loss.cpu().detach().numpy())

            # save model
            save_path = os.path.join(save_dir, f"checkpoint{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": blend_model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "loss": loss,
                },
                save_path,
            )

        # plot loss
        plt.plot(np.arange(len(loss_record)), loss_record, label="Loss")
        loss_path = os.path.join(model_dir, "loss.png")
        plt.savefig(loss_path)
        plt.gcf().clear()
        plt.plot(np.arange(len(loss_record[20:])), loss_record[20:], label="Loss")
        loss_path = os.path.join(model_dir, "loss_skip.png")
        plt.savefig(loss_path)
        plt.gcf().clear()

        # make train and test inferences
        blend_model.eval()
        if training_set == "_ms8":
            train_beg, train_end = 0, params["nb_train"]
            make_inference(
                blend_model,
                device,
                set_dir_ms8,
                train_beg,
                train_end,
                labels_ms8,
                preprocess_type,
                model_dir,
                "train_ms8",
            )

            test_beg, test_end = (
                params["nb_train"],
                params["nb_train"] + params["nb_test"],
            )
            make_inference(
                blend_model,
                device,
                set_dir_ms8,
                test_beg,
                test_end,
                labels_ms8,
                preprocess_type,
                model_dir,
                "test_ms8",
            )

            beg_ms4, end_ms4 = 0, params["nb_train"] + params["nb_test"]
            make_inference(
                blend_model,
                device,
                set_dir_ms4,
                beg_ms4,
                end_ms4,
                labels_ms4,
                preprocess_type,
                model_dir,
                "test_ms4",
            )

        elif training_set == "_ms4":
            train_beg, train_end = 0, params["nb_train"]
            make_inference(
                blend_model,
                device,
                set_dir_ms4,
                train_beg,
                train_end,
                labels_ms4,
                preprocess_type,
                model_dir,
                "train_ms4",
            )

            test_beg, test_end = (
                params["nb_train"],
                params["nb_train"] + params["nb_test"],
            )
            make_inference(
                blend_model,
                device,
                set_dir_ms4,
                test_beg,
                test_end,
                labels_ms4,
                preprocess_type,
                model_dir,
                "test_ms4",
            )

            beg_ms8, end_ms8 = 0, params["nb_train"] + params["nb_test"]
            make_inference(
                blend_model,
                device,
                set_dir_ms8,
                beg_ms8,
                end_ms8,
                labels_ms8,
                preprocess_type,
                model_dir,
                "test_ms8",
            )


if __name__ == "__main__":
    main()
