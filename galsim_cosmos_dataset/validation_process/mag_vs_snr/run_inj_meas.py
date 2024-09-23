import os
import sys

# corresponding tract and patches
# patches are divided to run in parallel and save time
tract = 9813

# data directories
data_dir = "../../../data"
suffix = "_release_run"
butler_dir = os.path.join(data_dir, "hsc_data")

# filters
filters = ["g", "r", "i", "z", "y"]
filter_s = ""
for filt in filters[:-1]:
    filter_s += f"HSC-{filt.upper()}^"
filter_s += f"HSC-{filters[-1].upper()}"


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

# rerun directory
rerun_name = f"pdr2_wide_magsnr_inj{suffix}"

for patch_s in patches:

    # multiBandDriver
    # runs with doDetection=True and hasFakes=True but it only makes the detectCoaddSources task
    cmd = f"multiBandDriver.py {butler_dir} --rerun={rerun_name}:{rerun_name}_1 --id tract={tract} patch={patch_s} filter={filter_s} -C multiBandDriver_config_inj1.py --cores=1 > logs_inj/logjcmulti1_{patch_s} 2> logs_inj/logjcmultierr1_{patch_s}"
    print(cmd)
    os.system(cmd)

    # multiBandDriver
    # runs without doDetection=True and without hasFakes=True to go through all the next tasks
    cmd = f"multiBandDriver.py {butler_dir} --rerun={rerun_name}_1:{rerun_name}_2 --id tract={tract} patch={patch_s} filter={filter_s} -C multiBandDriver_config_inj2.py --cores=2 --clobber-config > logs_inj/logjcmulti2_{patch_s} 2> logs_inj/logjcmultierr2_{patch_s}"
    print(cmd)
    os.system(cmd)
