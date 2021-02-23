# Welcome to IIB
Bo: I deleted the formal README and plan to write a tutorial here so that you can all learn how to start in this repo.

# How to start?
```shell
python -m domainbed.scripts.train \
--data_dir=/your_root_path/IIB/datasets
--algorithm=IIB
--dataset=RotatedMNIST
```
Use this script to download all datasets
```shell
python -m domainbed.scripts.download \
       --data_dir=/your_root_path/IIB/datasets
```
Note that `hparams` should be in json format, and wrap all key values in double quotes with an additional slash.