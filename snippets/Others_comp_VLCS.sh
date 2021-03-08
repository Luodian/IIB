cd /rscratch/luodian_libo/DomainBed

root_dir=/rscratch/luodian_libo/DomainBed
output_dir=Others_VLCS_PACS_comp

python -m domainbed.scripts.sweep delete_incomplete --command_launcher multi_gpu \
--data_dir $root_dir/datasets \
--output_dir $root_dir/train_output/$output_dir \
--algorithm ARM MTL IRM MMD DANN CDANN --dataset VLCS PACS \
--n_trials 1 --n_hparams 1 --skip_confirmation --single_test_envs

python -m domainbed.scripts.sweep launch --command_launcher multi_gpu \
--data_dir $root_dir/datasets \
--output_dir $root_dir/train_output/$output_dir \
--algorithm ARM MTL IRM MMD DANN CDANN --dataset VLCS PACS \
--n_trials 1 --n_hparams 1 --skip_confirmation --single_test_envs