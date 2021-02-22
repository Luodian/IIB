# Welcome to IIB
Bo: I deleted the formal README and plan to write a tutorial here so that you can all learn how to start in this repo.

# How to start?
```shell
python script/train.py
--data_dir=/your_root_path/IIB/datasets
--algorithm=IIB
--dataset=RotatedMNIST
--hparams={\"lambda_inv_risks\":100,\"embedding_dim\":256}
```
Note that `hparams` should be in json format, and wrap all key values in double quotes with an additional slash.