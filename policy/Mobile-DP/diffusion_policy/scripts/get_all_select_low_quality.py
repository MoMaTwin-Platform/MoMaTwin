import os
import json
import yaml

def has_select_low_quality_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return 'select_low_quality' in data and len(data['select_low_quality']) > 0

def solve(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        dataset = yaml.safe_load(f)
    for item in dataset['dataset_path']:
        # print(item['path'], flush=True)
        report_path = os.path.join(item['path'], 'report.json')
        if os.path.exists(report_path) and has_select_low_quality_data(report_path):
            print(f'{report_path}')

solve('/home/ganruyi/code/diffusion_policy/diffusion_policy/config/task/multitask_pickplace_3view_all.yaml')