import os

# tract and patches to inject sources in
# patches should correspond to what have been prepared with prepare_injection.py
tract = "9813"
patches = [
    "4,1",
    "4,2",
    "4,3",
    "4,4",
    "4,5",
    "4,6",
    "4,7",
    "4,8",
    "5,1",
    "5,2",
    "5,3",
    "5,4",
    "5,5",
    "5,6",
    "5,7",
    "5,8",
]

# filters
filters = ["g", "r", "i", "z", "y"]

# data directories
data_dir = "../../../data"
suffix = "_release_run"
butler_dir = os.path.join(data_dir, "hsc_data")
rerun_name = "pdr2_wide"

# run injection task
inject_dir = os.path.join(data_dir, f"for_magsnr_source_injection{suffix}")
for filt in filters:
    for patch_s in patches:
        cat_name = os.path.join(inject_dir, f"{tract}_{patch_s}.fits")

        cmd = f"insertFakes.py {butler_dir} --rerun={rerun_name}:{rerun_name}_magsnr_inj{suffix} --id tract={tract} patch={patch_s} filter=HSC-{filt.upper()} --config fakeType={cat_name} insertImages=True --clobber-config --clobber-versions"
        print(cmd)
        os.system(cmd)
