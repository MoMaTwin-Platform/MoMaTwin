import os
import json

def read_json_files(directories):
    data = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file == 'labelData_aug_new.json':
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        content = json.load(f)
                        for key, value in content.items():
                            item = {
                                'uid': key,
                                # 'instruction': value[0]['totalDesc'][0]
                                'instruction': value[0]
                            }
                            data.append(item)
    return data

def agumentation_files(directories):
    data = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file == 'labelData.json':
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        content = json.load(f)
                        for key, value in content.items():
                            item = {
                                'uid': key,
                                # 'instruction': value[0]['totalDesc'][0]
                                'instruction': value[0]
                            }
                            data.append(item)
    return data

def save_to_file(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str + '\n')

def main():
    input_directories = [
        "/x2robot/zhengwei/10000/20240429-pickplace_cups",
    "/x2robot/zhengwei/10000/20240430-cup-other",
    "/x2robot/zhengwei/10000/20240430-cup-addition",
    "/x2robot/zhengwei/10000/20240501-make-dish-cup-sameColour",
    "/x2robot/zhengwei/10000/20240502-pick_up-cup",
    "/x2robot/zhengwei/10001/20240506-pick_up-cup",
    "/x2robot/zhengwei/10000/20240506-take-cup",
    "/x2robot/zhengwei/10000/20240511-pick_up-cup-recover",
    "/x2robot/zhengwei/10000/20240521-pick_up-cup",
    "/x2robot/zhengwei/10001/20240529-take-cup",
    "/x2robot/zhengwei/10001/20240614-take-cup",
    "/x2robot/zhengwei/10002/20240614-take-cup",
    "/x2robot/zhengwei/10001/20240614-pick_up-spoon",
    "/x2robot/zhengwei/10001/20240614-pick_up-tape",
    "/x2robot/zhengwei/10001/20240614-pick_up-drink",
    "/x2robot/zhengwei/10001/20240614-pick_up-sponge",
    "/x2robot/zhengwei/10002/20240617-pick_up-item",
    "/x2robot/zhengwei/10001/20240624-pick_up-item",
    "/x2robot/zhengwei/10002/20240624-take-item",
    "/x2robot/zhengwei/10001/20240703-pick_up-item"
    ]
    output_file = '/x2robot/Data/text_augumentation_by_internvl/pickup_item_by_cc_0705.json'

    data = read_json_files(input_directories)
    save_to_file(data, output_file)

    print(f'Data saved to {output_file}')

if __name__ == "__main__":
    main()