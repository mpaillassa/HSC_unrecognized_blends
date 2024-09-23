import os


def main():

    bands = ["G", "R", "I", "Z", "Y"]
    tract = "9813"
    nb_patches = 18
    data_dir = "../../../data"
    options = f"-r --no-parent -nc --no-check-certificate --directory-prefix={data_dir}"

    survey = "pdr2_dud"  # , "pdr2_wide"]

    other_stuff = [
        "BrightObjectMasks/9813/",
        "BrightObjectMasks/9812/",
        "NewBrightObjectMasks/",
        "skyMap.pickle",
    ]  # , "skyMap.pickle.hscPipe6.5.1"]

    for stuff in other_stuff:
        cmd = f"wget https://hsc-release.mtk.nao.ac.jp/archive/filetree/{survey}/deepCoadd/{stuff} {options}"
        os.system(cmd)

    for band in bands:
        for patch in range(36, 36 + nb_patches):
            i1, i2 = patch // 9, patch % 9
            patch_s = f"{i1},{i2}"

            deepCoadd_dir = f"https://hsc-release.mtk.nao.ac.jp/archive/filetree/{survey}/deepCoadd/HSC-{band}/{tract}/{patch_s}/"
            cmd = f"wget {deepCoadd_dir} {options}"
            os.system(cmd)

            deepCoadd_file1 = f"https://hsc-release.mtk.nao.ac.jp/archive/filetree/{survey}/deepCoadd/HSC-{band}/{tract}/{patch_s}.fits"
            cmd = f"wget {deepCoadd_file1} {options}"
            os.system(cmd)

            deepCoadd_file2 = f"https://hsc-release.mtk.nao.ac.jp/archive/filetree/{survey}/deepCoadd/HSC-{band}/{tract}/{patch_s}_nImage.fits"
            cmd = f"wget {deepCoadd_file2} {options}"
            os.system(cmd)

            deepCoadd_results_dir = f"https://hsc-release.mtk.nao.ac.jp/archive/filetree/{survey}/deepCoadd-results/HSC-{band}/{tract}/{patch_s}/"
            cmd = f"wget {deepCoadd_results_dir} {options}"
            os.system(cmd)


if __name__ == "__main__":
    main()
