def IIB():
    print('jobs:')
    formatted_string = "  - name: domainbed_{algo}v2_{dataset}\n" \
                       "    sku: G1\n" \
                       "    sku_count: 1\n" \
                       "    aml_mpirun:\n" \
                       "      process_count_per_node: 1\n" \
                       "      communicator: \"OpenMpi\"\n" \
                       "    command:\n" \
                       "      - python -m domainbed.scripts.sweep --data_dir=$$PT_DATA_DIR --algorithm {algo} --datasets {dataset} --output_dir $$AMLT_OUTPUT_DIR --command_launcher local --skip_confirmation"
    for algo in ['LIRR']:
        for dataset in ['ColoredmMNIST', 'RotatedMNIST', 'VLCS', 'PACS', 'OfficeHome']:
            print(formatted_string.format(algo=algo, dataset=dataset))


def pruner():
    print('jobs:')
    formatted_string = "  - name: cifar100_prune_{model}_{algo}\n" \
                       "    sku: G1\n" \
                       "    sku_count: 1\n" \
                       "    aml_mpirun:\n" \
                       "      process_count_per_node: 1\n" \
                       "      communicator: \"OpenMpi\"\n" \
                       "    command:\n" \
                       "      - python basic_pruners.py --dataset cifar100 --model {model} --pruner {algo} --pretrain-epochs 160 --fine-tune-epochs 160 --speed-up --pretrained-model-dir ./experiment_data/pretrain_cifar100_{model}.pth --enable-trace 2>&1 | tee $$AMLT_OUTPUT_DIR/pruning_cifar100_{model}_{algo}.txt"

    for algo in ['l1filter', 'l2filter', 'fpgm', 'apoz', 'mean_activation', 'taylorfo']:
        for model in ["resnet18", "resnet34", "resnet50", "vgg16", "vgg19", "mobilenet_v1", "mobilenet_v2"]:
            print(formatted_string.format(algo=algo, model=model))


pruner()
