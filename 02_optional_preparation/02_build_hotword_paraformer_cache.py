# -*- coding: utf-8 -*-
# =========================================================
# 数据准备：最终热词 Paraformer 全量缓存
# 功能：写入最终领域热词，并补齐正/负样本 Paraformer 识别缓存。
# =========================================================

import json
import re
from pathlib import Path

import torch
from funasr import AutoModel
from tqdm import tqdm


PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data" / "testA"
OUTPUT_DIR = PROJECT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POS_JSONL = DATA_DIR / "pos.jsonl"
NEG_JSONL = DATA_DIR / "neg.jsonl"

# Step22 已经跑好的 1364 条正样本最优热词缓存
BEST_POS_CACHE = OUTPUT_DIR / "step22_core_strict_plus_temp16_30.jsonl"

# 新生成：1838 条正+负全量最优热词缓存
BEST_FULL_CACHE = OUTPUT_DIR / "asr_hotword_best_full_cache.jsonl"

# Step18 会优先读取这个文件
HOTWORDS_TXT = OUTPUT_DIR / "hotwords_clean.txt"


BEST_HOTWORDS = [
    "空调", "制冷", "制热", "除湿", "抽湿", "新风", "送风",
    "自动模式", "睡眠模式", "节能模式", "防直吹",
    "上下扫风", "左右扫风", "扫风",
    "风速", "风量", "风向", "出风口",
    "温度", "控温", "显示屏",
    "灯光", "亮度", "色温", "冷色调", "暖色调",
    "窗帘", "客厅", "厨房",
    "油烟机", "烟机", "滤网",
    "调高", "调低", "调大", "调小",
    "十六度", "十七度", "十八度", "十九度",
    "二十度", "二十一度", "二十二度", "二十三度", "二十四度",
    "二十五度", "二十六度", "二十七度", "二十八度", "二十九度",
    "三十度",
]


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(
        r"[，。！？、；：“”‘’（）()【】\[\],.!?;:'\"—-]",
        "",
        text,
    )
    return text


def normalize_path(path):

    text = str(path).strip().replace("\\", "/").lower()

    marker = "/data/testa/"

    if marker in text:
        return text.split(marker, 1)[1]

    return text


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_cache(path):
    cache = {}
    if not path.exists():
        return cache

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = item.get("command_key")

            if key:
                key = normalize_path(key)
            else:
                raw_path = item.get("command_path")

                if not raw_path:
                    continue

                key = normalize_path(raw_path)

            cache[key] = {
                "split": item.get("split", ""),
                "command_key": key,
                "command_path": item.get("command_path", ""),
                "text": normalize_text(item.get("text")),
            }

    return cache


for required in [POS_JSONL, NEG_JSONL, BEST_POS_CACHE]:
    if not required.exists():
        raise FileNotFoundError(f"缺少文件：{required}")


# ==================================================
# 1. 写正式热词文件
# ==================================================

HOTWORDS_TXT.write_text(
    "\n".join(BEST_HOTWORDS) + "\n",
    encoding="utf-8",
)

print("正式热词已写入：", HOTWORDS_TXT)
print("热词数量：", len(BEST_HOTWORDS))


# ==================================================
# 2. 读取数据
# ==================================================

pos_items = read_jsonl(POS_JSONL)
neg_items = read_jsonl(NEG_JSONL)

print("正样本：", len(pos_items))
print("负样本：", len(neg_items))


# ==================================================
# 3. 读取 Step22 最优正样本缓存
# ==================================================

best_pos_cache = load_cache(BEST_POS_CACHE)

print("Step22最优正样本缓存：", len(best_pos_cache))

if len(best_pos_cache) < len(pos_items):
    raise RuntimeError(
        f"最优正样本缓存不完整："
        f"{len(best_pos_cache)} / {len(pos_items)}"
    )


# ==================================================
# 4. 读取/初始化新的全量缓存
# ==================================================

full_cache = load_cache(BEST_FULL_CACHE)

# 把 1364 条正样本缓存复用进来
new_pos_count = 0

with BEST_FULL_CACHE.open("a", encoding="utf-8") as out:
    for item in pos_items:
        command_path = DATA_DIR / item["识别音频"]
        key = normalize_path(command_path)

        if key in full_cache:
            continue

        if key not in best_pos_cache:
            raise RuntimeError(f"正样本缺少Step22缓存：{command_path}")

        src = best_pos_cache[key]

        record = {
            "split": "pos",
            "command_key": key,
            "command_path": str(command_path),
            "text": src["text"],
        }

        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        full_cache[key] = record
        new_pos_count += 1

print("本次导入正样本：", new_pos_count)
print("当前全量缓存：", len(full_cache))


# ==================================================
# 5. 找出需要跑的负样本
# ==================================================

neg_samples = []

for item in neg_items:
    command_path = DATA_DIR / item["识别音频"]

    if not command_path.exists():
        raise FileNotFoundError(f"负样本音频不存在：{command_path}")

    key = normalize_path(command_path)

    neg_samples.append(
        {
            "command_path": command_path,
            "command_key": key,
        }
    )

missing_neg = [
    sample
    for sample in neg_samples
    if sample["command_key"] not in full_cache
]

print("本次需要识别的负样本：", len(missing_neg))


# ==================================================
# 6. 只跑缺失负样本
# ==================================================

if missing_neg:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("\n加载Paraformer……")
    print("设备：", device)

    model = AutoModel(
        model="paraformer-zh",
        device=device,
        disable_update=True,
    )

    hotword_string = " ".join(BEST_HOTWORDS)

    with BEST_FULL_CACHE.open("a", encoding="utf-8") as out:
        for sample in tqdm(
            missing_neg,
            desc="Step22b 最优热词负样本",
        ):
            with torch.inference_mode():
                result = model.generate(
                    input=str(sample["command_path"]),
                    hotword=hotword_string,
                )

            if (
                result
                and isinstance(result, list)
                and isinstance(result[0], dict)
            ):
                text = normalize_text(result[0].get("text", ""))
            else:
                text = ""

            record = {
                "split": "neg",
                "command_key": sample["command_key"],
                "command_path": str(sample["command_path"]),
                "text": text,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            full_cache[sample["command_key"]] = record


# ==================================================
# 7. 最终检查
# ==================================================

final_cache = load_cache(BEST_FULL_CACHE)

pos_keys = {
    normalize_path(DATA_DIR / item["识别音频"])
    for item in pos_items
}
neg_keys = {
    normalize_path(DATA_DIR / item["识别音频"])
    for item in neg_items
}

pos_found = sum(key in final_cache for key in pos_keys)
neg_found = sum(key in final_cache for key in neg_keys)

print("\n" + "=" * 60)
print("Step22b 最优热词全量缓存结果")
print("=" * 60)
print("正样本缓存：", pos_found, "/", len(pos_keys))
print("负样本缓存：", neg_found, "/", len(neg_keys))
print("总缓存：", len(final_cache))
print("目标总数：", len(pos_keys) + len(neg_keys))
print("输出文件：", BEST_FULL_CACHE)
print("正式热词：", HOTWORDS_TXT)

if pos_found != len(pos_keys) or neg_found != len(neg_keys):
    raise RuntimeError("全量缓存仍不完整，请重新运行本脚本继续补齐。")

print("\n成功：现在可以重新训练 Step17 / Step17b。")