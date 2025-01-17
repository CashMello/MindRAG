import os
import csv
import json
import requests
import re



OLLAMA_API_URL = "http://localhost:11434/api/generate"

def extract_json(text):
    """
    从文本中提取 JSON 部分。
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def evaluate_answers(query, answer1, answer2):

    # 构造评估提示
    system_prompt = """
    ---Role---
    You are an expert tasked with evaluating two answers to the same question based on three criteria: Comprehensiveness, Diversity, and Empowerment.

    ---Goal---
    Evaluate two answers to the same question based on the following criteria:
    1. Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
    2. Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
    3. Empowerment: How well does the answer help the reader understand and make informed judgments about the topic?

    For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. If Answer 1 and Answer is same output None. Then, select an overall winner based on these three categories.
    """

    user_prompt = f"""
    Here is the question: {query}
    Here are the two answers:
    Answer 1: {answer1}
    Answer 2: {answer2}

    Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

    Output your evaluation in the following strict JSON format:
    {{
        "Comprehensiveness": {{"Winner": "[Answer 1 or Answer 2 or None]", "Explanation": "[Provide explanation here]"}},
        "Diversity": {{"Winner": "[Answer 1 or Answer 2 or None]", "Explanation": "[Provide explanation here]"}},
        "Empowerment": {{"Winner": "[Answer 1 or Answer 2 or None]", "Explanation": "[Provide explanation here]"}},
        "Overall Winner": {{"Winner": "[Answer 1 or Answer 2 or None]", "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"}}
    }}
    """

    # 构造 Ollama 的请求数据
    payload = {
        "model": "qwen2.5:72b",
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "stream": False,
        "max_tokens": 10000,
        "temperature": 0.3
    }

    # 调用 Ollama 的 API
    response = requests.post(OLLAMA_API_URL, json=payload)
    print("Status code:", response.status_code)
    print("Response content:", response.text)

    if response.status_code != 200:
        raise Exception(f"Ollama API request failed with status code {response.status_code}: {response.text}")

    # 解析返回的结果
    evaluation_result = response.json()["response"].strip()
    #print("Raw evaluation result:", evaluation_result)

    # 提取 JSON 部分
    json_str = extract_json(evaluation_result)
    if json_str:
        return json.loads(json_str)
    else:
        raise ValueError("Failed to extract JSON from model response.")

def evaluate_csv(file_path):
    """
    从 CSV 文件中读取数据并评估答案。
    """
    results = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            query = row["query"]
            answer1 = row["response"]
            answer2 = row["response_graph"]
            try:
                evaluation = evaluate_answers(query, answer1, answer2)
                results.append({
                    "query": query,
                    "answer1": answer1,
                    "answer2": answer2,
                    "evaluation": evaluation
                })
            except Exception as e:
                print(f"Error evaluating query: {query}. Error: {e}")
    return results

def count_wins(evaluation_results):
    """
    统计每个指标上两个答案的获胜次数。
    """
    wins = {
        "Answer 1": {"Comprehensiveness": 0, "Diversity": 0, "Empowerment": 0, "Overall Winner": 0},
        "Answer 2": {"Comprehensiveness": 0, "Diversity": 0, "Empowerment": 0, "Overall Winner": 0},
        "None": {"Comprehensiveness": 0, "Diversity": 0, "Empowerment": 0, "Overall Winner": 0}
    }

    for result in evaluation_results:
        evaluation = result["evaluation"]
        for criterion, details in evaluation.items():
            if criterion == "Overall Winner":
                wins[details["Winner"]][criterion] += 1
            else:
                wins[details["Winner"]][criterion] += 1

    return wins

# 示例 CSV 文件路径
csv_file_path = "response_hyper_test.csv"  # 替换为你的 CSV 文件路径

# 评估答案
evaluation_results = evaluate_csv(csv_file_path)

# 打印评估结果
for result in evaluation_results:
    print(json.dumps(result, indent=4, ensure_ascii=False))

# 统计获胜次数
win_counts = count_wins(evaluation_results)
print("\nWinning counts:")
print(json.dumps(win_counts, indent=4, ensure_ascii=False))
