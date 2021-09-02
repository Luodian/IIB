def IIB():
    print('jobs:')
    formatted_string = "  - name: domainbed_{algo}v6_{dataset}_{lmd_beta}_{lmd_risk}_{bn}\n" \
                       "    sku: G8\n" \
                       "    sku_count: 1\n" \
                       "    aml_mpirun:\n" \
                       "      process_count_per_node: 1\n" \
                       "      communicator: \"OpenMpi\"\n" \
                       "    command:\n" \
                       "      - python -m domainbed.scripts.sweep launch --data_dir=$$PT_DATA_DIR --algorithm {algo} --datasets {dataset} --output_dir $$AMLT_OUTPUT_DIR --command_launcher multi_gpu --lambda_beta {lmd_beta} --lambda_inv_risks {lmd_risk} --enable_bn {bn} --skip_confirmation"
    for algo in ['IIB']:
        for dataset in ['TerraIncognita', 'SVIRO', 'WILDSCamelyon', 'WILDSFMoW']:
            for lmd_risk in [10]:
                for lmd_beta in ['1e-4']:
                    for bn in ['True']:
                        print(formatted_string.format(algo=algo, dataset=dataset, lmd_risk=lmd_risk, lmd_beta=lmd_beta, bn=bn))


IIB()
