# -*- coding: utf-8 -*-
# =========================================================
# 数据准备：领域热词候选自动生成
# 功能：从正样本标签中分词、统计词频并生成候选热词。
# 说明：这是热词候选生成阶段，不是最终推理必须文件。
# =========================================================

import csv
import json
import re
from collections import Counter
from pathlib import Path

import jieba


# ==================================================
# 1. 基础路径
# ==================================================

project_path = Path(__file__).resolve().parents[1]
data_path = project_path / "data" / "testA"
output_path = project_path / "output"

pos_jsonl_path = data_path / "pos.jsonl"
hotword_txt_path = output_path / "hotwords.txt"
hotword_csv_path = output_path / "hotword_statistics.csv"

output_path.mkdir(exist_ok=True)


# ==================================================
# 2. 基础设置
# ==================================================

# 最多保留多少个热词
max_hotwords = 100

# 至少在标签中出现多少次
minimum_count = 3

# 过滤无实际领域意义的常用词
stopwords = {
    "一下",
    "现在",
    "然后",
    "帮我",
    "给我",
    "请你",
    "我想",
    "我要",
    "需要",
    "可以",
    "进行",
    "这个",
    "那个",
    "一个",
    "已经",
    "还是",
    "就是",
    "不要",
    "怎么",
    "什么",
}


# ==================================================
# 3. 文本标准化
# ==================================================

def normalize_text(text):
    if text is None:
        return ""

    text = str(text).strip()

    # 去掉所有空白
    text = re.sub(r"\s+", "", text)

    # 去掉常见标点
    text = re.sub(
        r"[，。！？、；：“”‘’（）()【】\[\],.!?;:'\"—-]",
        "",
        text,
    )

    return text


# ==================================================
# 4. 判断是否为合适的中文热词
# ==================================================

def is_valid_hotword(word):
    word = word.strip()

    if word in stopwords:
        return False

    # 热词长度控制在2到8个字符
    if not 2 <= len(word) <= 8:
        return False

    # 至少包含一个中文字符
    if not re.search(r"[\u4e00-\u9fff]", word):
        return False

    return True


# ==================================================
# 5. 读取全部正样本标签
# ==================================================

if not pos_jsonl_path.exists():
    raise FileNotFoundError(f"找不到文件：{pos_jsonl_path}")

labels = []

with pos_jsonl_path.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        line = line.strip()

        if not line:
            continue

        item = json.loads(line)
        label = normalize_text(item.get("识别文本"))

        if label:
            labels.append(label)

print("读取到的正样本标签数量：", len(labels))


# ==================================================
# 6. 使用jieba切分并统计词频
# ==================================================

word_counter = Counter()

for label in labels:
    words = jieba.lcut(label)

    for word in words:
        word = word.strip()

        if is_valid_hotword(word):
            word_counter[word] += 1


# ==================================================
# 7. 过滤低频词并选取前100个
# ==================================================

candidate_hotwords = [
    (word, count)
    for word, count in word_counter.items()
    if count >= minimum_count
]

candidate_hotwords.sort(
    key=lambda item: (
        -item[1],
        -len(item[0]),
        item[0],
    )
)

selected_hotwords = candidate_hotwords[:max_hotwords]


# ==================================================
# 8. 显示热词
# ==================================================

print("\n==============================")
print("自动生成的热词")
print("==============================")

for index, (word, count) in enumerate(
    selected_hotwords,
    start=1,
):
    print(
        f"{index:03d}. "
        f"{word}，出现次数：{count}"
    )


# ==================================================
# 9. 保存hotwords.txt
# ==================================================

with hotword_txt_path.open(
    "w",
    encoding="utf-8",
) as file:

    for word, _ in selected_hotwords:
        file.write(word + "\n")


# ==================================================
# 10. 保存完整统计CSV
# ==================================================

with hotword_csv_path.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "rank",
            "hotword",
            "count",
        ],
    )

    writer.writeheader()

    for index, (word, count) in enumerate(
        selected_hotwords,
        start=1,
    ):
        writer.writerow(
            {
                "rank": index,
                "hotword": word,
                "count": count,
            }
        )


print("\n热词生成完成：")
print("热词文本：", hotword_txt_path)
print("统计表格：", hotword_csv_path)
print("最终热词数量：", len(selected_hotwords))