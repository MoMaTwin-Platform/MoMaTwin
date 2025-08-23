# %%
# python3
# Please install OpenAI SDK first：`pip3 install openai`
from openai import OpenAI
import os
import time

client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url="https://api.deepseek.com")

system='''注意我现在在给机械臂下达指令，你是一个翻译者，要求把我给机械臂的指令翻译成英文，并且在不改变原指令意思的前提下进行改写，要求：
1. 如果给机械臂的是中文指令，生成4条英文简洁的指令,不能改变原来的意思。
2. 如果遇到一些品牌名字，不要翻译，直接省略品牌部分，比如农夫山泉直接翻译成水瓶的英文，君乐宝酸奶翻译成酸奶英文等。
3. 如果同时遇到用材质、颜色、形状描述一个物品，保留颜色即可，材质、形状直接省略。
4. 不要生成序号，4条翻译指令生成4行。
5. 如果给你的已经是英文了，你直接生成4条改写的简洁的英文指令，不改变指令的意思。'''
prompt = '''用右臂把桌面上农夫山泉放到青色陶瓷盘上。'''
prompt = '''Place the bottle on the blue ceramic plate using your right arm.'''

def evol_instructions_with_deepseekchat(system, prompt, max_retries=2):
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=1.1,
            )
            content = response.choices[0].message.content
            print(content)
            contents = content.split('\n')
            contents = [c for c in contents if c != '']
            assert len(contents) == 4
            return contents
        # except (APIError, RateLimitError, ServiceUnavailableError) as e:
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"Error occurred: {str(e)}. Retrying in 3 seconds... (Attempt {retry_count}/{max_retries})")
                time.sleep(3)
            else:
                print(f"Error occurred: {str(e)}. Max retries exceeded. Exiting.")
                return []
                # raise


# evol_instructions_with_deepseekchat(system=system, prompt=prompt)

# %%
import os
import json
import yaml

def process_labeldata(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    res = []
    for key, value in data.items():
        uid = key.replace('%40', '@')
        instruction_label = value[0]['totalDesc']
        instructions = evol_instructions_with_deepseekchat(system=system, prompt=instruction_label)
        res.append({'uid': uid, 'instruction-label': [instruction_label], 'instruction': instructions})
    return res

def solve(yaml_path, result_file='result.json'):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        dataset = yaml.safe_load(f)
    # 检查结果文件是否存在,如果存在则删除
    if os.path.exists(result_file):
        os.remove(result_file)
    
    result = []
    for item in dataset['dataset_path']:
        # if item['path'] != '/x2robot/zhengwei/10002/20240617-pick_up-item':
        #     continue
        print(item['path'], flush=True)
        labeldata_path = os.path.join(item['path'], 'labelData.json')
        if os.path.exists(labeldata_path):
            item_data = process_labeldata(labeldata_path)
            # 读取已有的结果数据
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
            else:
                result = []
            
            # 将新的结果数据添加到已有数据中
            result.extend(item_data)
            
            # 将更新后的结果数据写入文件
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            print(f'{labeldata_path} is not exists!!!')
    
    with open(result_file, 'w', encoding='utf-8') as f:
        # json.dump(result, f, ensure_ascii=False)
        json.dump(result, f, ensure_ascii=False, indent=2)

def solve_fail(result_file='result.json', new_result_file='result.json'):
    assert os.path.exists(result_file), f'{result_file} not exists!!!'
    with open(result_file, 'r', encoding='utf-8') as f:
        result = json.load(f)
        all_result = []
        for item in result:
            if len(item['instruction']) != 4:
                instructions = evol_instructions_with_deepseekchat(system=system, prompt=item['instruction-label'][0], max_retries=10)
                all_result.append({'uid': item["uid"], 'instruction-label': item["instruction-label"], 'instruction': instructions})
            else:
                all_result.append(item)
    with open(new_result_file, 'w', encoding='utf-8') as f:
        json.dump(all_result, f, ensure_ascii=False, indent=2)
        
# solve('/home/ganruyi/code/diffusion_policy/diffusion_policy/config/task/multitask_pickplace_3view_all_right.yaml')
solve_fail(result_file='/x2robot/Data/text_augumentation_by_internvl/pickup_item_by_deepseek_v2.json', new_result_file='/x2robot/Data/text_augumentation_by_internvl/pickup_item_by_deepseek_v3.json')