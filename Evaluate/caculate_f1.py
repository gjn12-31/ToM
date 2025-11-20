import json
import string
from collections import Counter
import re


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Calculate F1 score between prediction and ground truth."""
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truths) -> float:
    # 确保 prediction 是一个列表
    if isinstance(prediction, str):
        prediction = [prediction]
    elif prediction is None:
        prediction = []

    # 处理每个预测值
    processed_predictions = []
    for pred in prediction:
        if pred is None or pred.lower() == "none":
            processed_predictions.append("")
        else:
            # 去除 "**Answer**:" 的前缀并去掉首尾空格
            processed_predictions.append(re.sub(r'^\*\*Answer\*\*:', '', pred).strip())
            print(processed_predictions)

    # 确保 ground_truths 是一个列表
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    # 计算 F1 分数
    f1_scores = []
    for ground_truth in ground_truths:
        for pred in processed_predictions:  # 遍历每个预测值
            # Normalize texts
            normalized_prediction = normalize_answer(pred)
            normalized_ground_truth = normalize_answer(ground_truth)

            # Split into tokens
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()

            # 计算每对 prediction 和 ground_truth 的 F1 分数
            f1_scores.append(f1_score(prediction_tokens, ground_truth_tokens))

    # 返回最大 F1 分数
    return max(f1_scores) if f1_scores else 0


def evaluate_qa_file(file_path: str):
    """
    Evaluate QA predictions from a JSON file.
    File should contain list of objects with 'query', 'answer', and 'prediction' fields.
    Skip empty predictions when calculating scores.
    """
    # Read JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)[0:50]

    # Calculate scores
    scores = []
    total_samples = len(data)
    skipped_samples = 0

    for item in data:
        # 检查prediction是否为空
        if not item.get('prediction') or (isinstance(item['prediction'], str) and not item['prediction'].strip()):
            skipped_samples += 1
            continue

        score = qa_f1_score(item['prediction'], item['answer'])
        scores.append(score)

    # Calculate average score, excluding skipped samples
    valid_samples = total_samples - skipped_samples
    avg_score = sum(scores) / valid_samples if scores else 0

    # Print detailed results
    print(f"\nTotal number of samples: {total_samples}")
    print(f"Number of valid samples: {valid_samples}")
    print(f"Number of skipped samples: {skipped_samples}")
    print(f"Average F1 Score: {avg_score:.4f}")


if __name__ == "__main__":
    # 指定输入文件路径
    file_path = ""
    evaluate_qa_file(file_path)