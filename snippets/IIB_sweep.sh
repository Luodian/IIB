cd /rscratch/luodian_libo/DomainBed

python -m domainbed.scripts.sweep delete_incomplete --command_launcher multi_gpu \
--data_dir=/rscratch/luodian_libo/DomainBed/datasets \
--output_dir /rscratch/luodian_libo/DomainBed/train_output/IIB \
--algorithm IIB --dataset RotatedMNIST ColoredMNIST VLCS PACS OfficeHome TerraIncognita DomainNet SVIRO --hparams={\"lambda_inv_risks\":100} \
--n_trials 1 --n_hparams 1 --skip_confirmation --single_test_envs

python -m domainbed.scripts.sweep launch --command_launcher multi_gpu \
--data_dir=/rscratch/luodian_libo/DomainBed/datasets \
--output_dir /rscratch/luodian_libo/DomainBed/train_output/IIB \
--algorithm IIB --dataset RotatedMNIST ColoredMNIST VLCS PACS OfficeHome TerraIncognita DomainNet SVIRO --hparams={\"lambda_inv_risks\":100} \
--n_trials 1 --n_hparams 1 --skip_confirmation --single_test_envs

