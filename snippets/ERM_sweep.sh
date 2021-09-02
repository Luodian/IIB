cd /home/v-boli4/teamdrive/users/drluodian/DomainBed

python -m domainbed.scripts.sweep delete_incomplete --command_launcher local \
--data_dir /home/v-boli4/teamdrive/msrashaiteamdrive/data \
--output_dir /home/v-boli4/teamdrive/users/drluodian/IIB/train_output/test \
--algorithm ERM --dataset ColoredMNIST \
--skip_confirmation --lambda_inv_risks 20

python -m domainbed.scripts.sweep launch --command_launcher local \
--data_dir /home/v-boli4/teamdrive/msrashaiteamdrive/data \
--output_dir /home/v-boli4/teamdrive/users/drluodian/IIB/train_output/test \
--algorithm ERM --dataset ColoredMNIST \
--skip_confirmation --lambda_inv_risks 20