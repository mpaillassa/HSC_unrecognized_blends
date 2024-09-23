import os

import galcheat

# some variables
nb_coadds = 197
suffix = "_release_run"
data_dir = "../../../data"
coadd_dir = os.path.join(
    data_dir, f"galsim_cosmos_dataset/validation_process/fake_coadds{suffix}"
)

# this is the butler repository that needs to be setup correctly
# see https://github.com/jcoupon/importExtData
butler_dir = os.path.join(data_dir, "my_data")
rerun_name = f"validation_process{suffix}"

# get filters
hsc_survey = galcheat.get_survey("HSC")
filters = hsc_survey.available_filters
filter_string = ""
for filt in filters[:-1]:
    filter_string += f"HSC-{filt.upper()}^"
filter_string += f"HSC-{filters[-1].upper()}"

# log directory
if not os.path.isdir("logs"):
    os.mkdir("logs")

# process every coadd
for coadd in range(nb_coadds):

    # emulateHscCoadd
    for filt in filters:
        imgname = os.path.join(coadd_dir, f"coadd_{filt}_{coadd:05d}.fits")
        varname = os.path.join(coadd_dir, f"var_{filt}_{coadd:05d}.fits")

        zp = hsc_survey.get_filter(filt).zeropoint.value

        command = f"emulateHscCoadd.py {butler_dir} --rerun={rerun_name} --id tract={coadd} patch=0,0 filter=HSC-{filt.upper()} --config imgInName={imgname} varInName={varname} weight=False mag0={zp} > logs/emullog_{coadd}_{filt} 2> logs/emuallogerr_{coadd}_{filt}"
        print(command)
        os.system(command)

    # detectCoaddSources
    command = f"detectCoaddSources.py {butler_dir} --rerun={rerun_name}:{rerun_name} --id tract={coadd} patch=0,0 filter={filter_string} > logs/detlog_{coadd} 2> logs/detlogerr_{coadd}"
    print(command)
    os.system(command)

    # multiBandDriver
    cmd = f"multiBandDriver.py {butler_dir} --rerun={rerun_name}:{rerun_name} --id tract={coadd} patch=0,0 filter={filter_string} -C multiBandDriver_config.py --cores=1 > logs/multilog_{coadd} 2> logs/multilogerr_{coadd}"
    print(cmd)
    os.system(cmd)
