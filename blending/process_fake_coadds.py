import os

import galcheat

###
suffix = "_release_run"
data_dir = "../data"
for_ml_training = False
if for_ml_training:
    coadd_dir = os.path.join(data_dir, f"blending/set_with_hst_ms8_fake_coadds{suffix}")
    rerun_name = "set_with_hst_ms8"
    log_dir = "logs_ms8"
else:
    coadd_dir = os.path.join(data_dir, f"blending/set_with_hst_ms4_fake_coadds{suffix}")
    rerun_name = "set_with_hst_ms4"
    log_dir = "logs_ms4"

if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

butler_dir = os.path.join(data_dir, "my_data")
nb_coadds = 132

hsc_survey = galcheat.get_survey("HSC")
filters = hsc_survey.available_filters
filter_string = ""
for filt in filters[:-1]:
    filter_string += f"HSC-{filt.upper()}^"
filter_string += f"HSC-{filters[-1].upper()}"

for coadd in range(nb_coadds):

    # emulateHscCoadd
    for filt in filters:
        imgname = os.path.join(coadd_dir, f"coadd_{filt}_{coadd:05d}.fits")
        varname = os.path.join(coadd_dir, f"var_{filt}_{coadd:05d}.fits")

        zp = hsc_survey.get_filter(filt).zeropoint.value

        command = f"emulateHscCoadd.py {butler_dir} --rerun={rerun_name} --id tract={coadd} patch=0,0 filter=HSC-{filt.upper()} --config imgInName={imgname} varInName={varname} weight=False mag0={zp} > {log_dir}/emullog_{coadd}_{filt} 2> {log_dir}/emuallogerr_{coadd}_{filt}"
        print(command)
        os.system(command)

    # detectCoaddSources
    command = f"detectCoaddSources.py {butler_dir} --rerun={rerun_name}:{rerun_name} --id tract={coadd} patch=0,0 filter={filter_string} > {log_dir}/detlog_{coadd} 2> {log_dir}/detlogerr_{coadd}"
    print(command)
    os.system(command)

    # multiBandDriver
    cmd = f"multiBandDriver.py {butler_dir} --rerun={rerun_name}:{rerun_name} --id tract={coadd} patch=0,0 filter={filter_string} -C multiBandDriver.py --cores=1 > {log_dir}/multilog_{coadd} 2> {log_dir}/multilogerr_{coadd}"
    print(cmd)
    os.system(cmd)
