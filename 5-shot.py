import os
import torch
import json
import string
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from metric.metrics import MetricAccuracy

# 设置环境变量，指定使用 GPU（可根据需要修改）
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# 检查是否可以使用 CUDA
if torch.cuda.is_available():
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    current_device = torch.cuda.current_device()
    print(f"Current GPU device ID: {current_device}")
    print(f"Current GPU device name: {torch.cuda.get_device_name(current_device)}")
else:
    print("CUDA is not available. Running on CPU.")

def load_baichuan_model():
    """
    加载 Baichuan 模型和 tokenizer。
    """
    cache_dir = "/home/chenbingxuan/models/Baichuan/snapshots"
    tokenizer = AutoTokenizer.from_pretrained(cache_dir, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cache_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(cache_dir)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return tokenizer, model

def format_question(task_meta, patient_data):
    """
    使用任务元数据格式化问题，将患者描述插入到问题结构中。
    """
    question = task_meta['question structure'].format(patient_description=patient_data['患者描述'])
    options = string.ascii_uppercase[:task_meta['option count']]
    option_str = '\n选项：'
    for i, option in enumerate(options):
        option_str += f'\n（{option}）{patient_data["options"][i]}'
    return question + option_str

def get_few_shot_examples(task_name):
    """
    获取指定任务的 5-shot 示例。
    """
    examples_file = Path('5-shot_instance/5-shot_examples.json')
    if not examples_file.exists():
        print(f"示例文件 {examples_file} 不存在。")
        return []
    with open(examples_file, 'r', encoding='utf-8') as f:
        examples_data = json.load(f)
    return examples_data.get(task_name, [])

def extract_option(response_str, options_list):
    """
    从模型的响应中提取选项标识。
    """
    # 尝试直接提取选项字母
    match_option = re.search(r'[\(（]([A-Z])[\)）]', response_str)
    if match_option:
        option = match_option.group(1)
        valid_options = [get_option_label(i) for i in range(len(options_list))]
        if option in valid_options:
            return option

    # 尝试从回答中提取答案文本
    patterns = [
        r'答案是\s*([\S\s]+?)。',
        r'为\s*([\S\s]+?)。',
        r'是\s*([\S\s]+?)。'
    ]
    for pattern in patterns:
        match_answer = re.search(pattern, response_str)
        if match_answer:
            answer_text = match_answer.group(1).strip('。，；,')
            # 将答案文本与选项列表进行匹配
            for idx, option_text in enumerate(options_list):
                if answer_text in option_text or option_text in answer_text:
                    return get_option_label(idx)

    # 在整个响应中搜索选项
    for idx, option_text in enumerate(options_list):
        if option_text in response_str:
            return get_option_label(idx)

    return '无法匹配'

def get_option_label(index):
    """
    根据索引生成选项字母。
    """
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if index < 26:
        return letters[index]
    else:
        return letters[index // 26 - 1] + letters[index % 26]

def plot_radar_chart(task2result, model_name, result_dir):
    """
    绘制模型在各个任务上的性能雷达图。
    """
    metrics = list(task2result.keys())
    performance = list(task2result.values())
    performance = [min(1, max(0, val)) for val in performance]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    performance += performance[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, performance, color='blue', alpha=0.25)
    ax.plot(angles, performance, color='blue', linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    plt.title(f'{model_name} Model Performance Evaluation', size=20, color='black', y=1.05)
    graph_dir = result_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    radar_chart_path = graph_dir / 'performance_radar_chart.png'
    plt.savefig(radar_chart_path)
    plt.show()

def main():
    """
    主函数，负责 5-shot 实验的流程。
    """
    # 加载模型
    tokenizer, model = load_baichuan_model()
    model_name = "Baichuan-13B"
    evaluation_type = '5-shot'

    # 定义要评估的任务
    tasks = ['BPlevel', 'stratification', 'singledrug(chemical)', 'singledrug(type)', 'combineddrugs']
    task_dir = 'tasks'  # 任务文件夹
    result_dir = 'result'  # 结果文件夹

    # 输入 system prompt
    system_prompt = input("请输入 system prompt：").strip()

    # 获取当前日期和时间，创建结果目录
    current_time = datetime.now().strftime('%m-%d_%H-%M')
    result_dir = Path(result_dir) / evaluation_type / model_name / current_time
    result_dir.mkdir(parents=True, exist_ok=True)

    # 保存 system prompt 到文件
    prompt_file = result_dir / 'system_prompt.txt'
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(f"System Prompt:\n{system_prompt}\n")

    task2result = {}  # 存储各任务的评估结果

    # 遍历每个任务
    for task_name in tasks:
        task_path = Path(task_dir, task_name)  # 任务路径
        task_meta = json.loads((task_path / 'config.json').read_text(encoding='utf-8'))
        task_data = json.loads((task_path / 'data.json').read_text(encoding='utf-8'))
        task_meta['system prompt'] = system_prompt  # 更新 system prompt

        # 获取 5-shot 示例
        examples = get_few_shot_examples(task_name)
        if not examples:
            print(f"任务 {task_name} 没有找到 5-shot 示例，跳过该任务。")
            continue

        processed_data = []  # 存储已处理的患者数据

        # 处理每个患者的数据（只处理前三个患者）
        for i, patient in enumerate(task_data[:3]):
            question = format_question(task_meta, patient)
            patient['question'] = question

            print('*' * 50, f'Task name:{task_name}, Patient {i+1}', '*' * 50)

            # 构建包含 5-shot 示例的输入
            few_shot_prompt = ""
            for ex in examples:
                few_shot_prompt += f"{ex['input']}\n{ex['output']}\n\n"

            # 添加当前问题
            few_shot_prompt += f"{question}"

            # 构建消息
            messages = [{"role": "system", "content": system_prompt}]
            messages.append({"role": "user", "content": few_shot_prompt})

            input_text = " ".join([message['content'] for message in messages])
            print("输入的问题: ", input_text)

            # 生成模型回答
            response = model.chat(tokenizer, [{"role": "user", "content": input_text}])
            patient['response'] = response
            # 提取答案
            final_extract = extract_option(response, patient['options'])
            patient['extract'] = final_extract
            print('模型回答：' + response + '\n' + '-' * 100)

            processed_data.append(patient)

        # 保存结果
        df = pd.DataFrame(processed_data)
        result_path = result_dir / 'response' / task_name
        result_path.mkdir(parents=True, exist_ok=True)
        # 保存任务数据为 JSON 文件
        Path(result_path / 'patient_response.json').write_text(
            json.dumps(processed_data, ensure_ascii=False, indent=4),
            encoding='utf-8'
        )
        # 保存任务数据为 Excel 文件
        df.to_excel(result_path / 'response.xlsx', index=False)

        # 评估结果
        metric = MetricAccuracy()
        for data_dict in processed_data:
            answer_id = data_dict['answer_id']
            answer = get_option_label(answer_id)
            data_dict['answer'] = answer
            extract = data_dict['extract']
            metric.compare(answer, extract)

        result = metric.get_result()
        (result_path / 'result.json').write_text(json.dumps({'accuracy': result}, ensure_ascii=False, indent=4), encoding='utf-8')
        task2result[task_name] = result

    # 保存整体结果
    (result_dir / 'result.json').write_text(json.dumps(task2result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(task2result)
    plot_radar_chart(task2result, model_name, result_dir)

if __name__ == "__main__":
    main()