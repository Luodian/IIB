root_dir=/rscratch/luodian_libo/DomainBed
output_dir=IIB_comp

cd $root_dir

python -m domainbed.scripts.sweep delete_incomplete --command_launcher multi_gpu \
--data_dir $root_dir/datasets \
--output_dir $root_dir/train_output/$output_dir \
--algorithm ERM IRM CORAL MMD IIB DANN CDANN --dataset ColoredMNIST RotatedMNIST VLCS PACS OfficeHome TerraIncognita \
--n_trials 3 --n_hparams 1 --skip_confirmation

python -m domainbed.scripts.sweep launch --command_launcher multi_gpu \
--data_dir $root_dir/datasets \
--output_dir $root_dir/train_output/$output_dir \
--algorithm ERM IRM CORAL MMD IIB DANN CDANN --dataset ColoredMNIST RotatedMNIST VLCS PACS OfficeHome TerraIncognita \
--n_trials 3 --n_hparams 1 --skip_confirmation