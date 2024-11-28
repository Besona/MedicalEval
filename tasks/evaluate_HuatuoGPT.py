import os
import torch
import json
import string
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

# 设置环境变量，指定使用 GPU 7
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def load_huatuo_model():
    # 加载 HuatuoGPT 模型和 tokenizer
    cache_dir = "/home/chenbingxuan/models/HuatuoGPT/snapshots"
    tokenizer = AutoTokenizer.from_pretrained(cache_dir, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(cache_dir, device_map="auto", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)

    # 设置生成配置
    model.generation_config = GenerationConfig.from_pretrained(cache_dir)

    # 启用 gradient checkpointing 以减少显存占用
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return tokenizer, model


# 定义问题格式化逻辑
def format_question(task_meta, patient_data):
    question = task_meta['question structure'].format(patient_description=patient_data['患者描述'])
    options = string.ascii_uppercase[:task_meta['option count']]
    option_str = '\n选项：'
    for i, option in enumerate(options):
        option_str += f'\n（{option}）{patient_data["options"][i]}'
    return question + option_str


# 构造对话消息
def get_messages(task_meta, question):
    messages = [
        {"role": "system", "content": task_meta['system prompt']},
        {"role": "user", "content": question}
    ]
    return messages


def main():
    # 加载 HuatuoGPT 模型
    tokenizer, model = load_huatuo_model()

    tasks = ['BPlevel', 'stratification', 'singledrug(chemical)', 'singledrug(type)', 'combineddrugs']
    task_dir = 'tasks'
    result_dir = 'result'
    evaluation_type = '0-shot'
    result_dir = Path(result_dir) / evaluation_type / "HuatuoGPT"

    for task_name in tasks:
        task_path = Path(task_dir, task_name)

        # 加载任务元数据和数据
        try:
            task_meta = json.loads((task_path / 'config.json').read_text(encoding='utf-8'))
            task_data = json.loads((task_path / 'data.json').read_text(encoding='utf-8'))
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue

        messages_list = []
        question_list = []

        # 处理每个患者的数据
        for patient in task_data:
            question = format_question(task_meta, patient)
            patient['question'] = question
            question_list.append(question)
            messages_list.append(get_messages(task_meta, question))

        print('*' * 50, f'Task name:{task_name}', '*' * 50)

        for i, messages in enumerate(messages_list):
            # 创建一个新的消息列表，包含用户的输入
            input_messages = []

            # 将问题内容添加为用户的输入
            input_text = " ".join([message['content'] for message in messages])
            input_messages.append({"role": "user", "content": input_text})

            # 打印输入的问题
            print("输入的问题: ", input_text)

            # 使用 HuatuoGPT 模型生成响应
            response = model.chat(tokenizer, input_messages)

            # 输出模型生成的回答
            response_text = response
            print("生成的答案: ", response_text)

            # 存储模型的回答到任务数据
            task_data[i]['response'] = response_text
            print('模型回答：' + response_text + '\n' + '-' * 100)

        # 保存结果
        df = pd.DataFrame(task_data)
        result_path = result_dir / 'response' / task_name
        result_path.mkdir(parents=True, exist_ok=True)
        Path(result_path / 'patient_response.json').write_text(json.dumps(task_data, ensure_ascii=False, indent=4),
                                                               encoding='utf-8')
        df.to_excel(result_path / 'response.xlsx', index=False)


if __name__ == "__main__":
    main()