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


IIB()
