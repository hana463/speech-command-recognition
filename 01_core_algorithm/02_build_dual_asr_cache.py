
import csv
import json
import re
import time
import unicodedata
from pathlib import Path

import editdistance
import joblib
import numpy as np
import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from scipy.sparse import csr_matrix, hstack
from tqdm import tqdm


# =========================================================
# Step31：构建SenseVoice全量1838缓存 + 三套双ASR候选缓存
#
# 输出：
# 1. asr_sensevoice_full_cache.jsonl
# 2. asr_dual_shorter_full_cache.jsonl
# 3. asr_dual_selector_t050_full_cache.jsonl
# 4. asr_dual_selector_best_full_cache.jsonl
# 5. asr_dual_selector_best_oof_policy_cache.jsonl
#
# 第5个文件：
# - 正样本使用Step30 OOF概率选结果，避免selector训练泄漏
# - 负样本使用最终selector
# 后续专门用于“门控策略OOF验证”
# =========================================================

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data" / "testA"
OUTPUT_DIR = PROJECT / "output"

POS_JSONL = DATA_DIR / "pos.jsonl"
NEG_JSONL = DATA_DIR / "neg.jsonl"

PARA_CACHE_PATH = OUTPUT_DIR / "asr_hotword_best_full_cache.jsonl"

# Step29已经把1364条正样本写进这个缓存
STEP29_SV_CACHE = OUTPUT_DIR / "step28_sensevoice_sample_cache.jsonl"

FULL_SV_CACHE = OUTPUT_DIR / "asr_sensevoice_full_cache.jsonl"

SELECTOR_MODEL = OUTPUT_DIR / "dual_asr_selector.joblib"
STEP30_OOF = OUTPUT_DIR / "step30_dual_asr_selector_oof.csv"
STEP18_DETAILS = OUTPUT_DIR / "final_testA_inference_details_optimized.csv"

SHORTER_CACHE = OUTPUT_DIR / "asr_dual_shorter_full_cache.jsonl"
T050_CACHE = OUTPUT_DIR / "asr_dual_selector_t050_full_cache.jsonl"
BEST_CACHE = OUTPUT_DIR / "asr_dual_selector_best_full_cache.jsonl"
OOF_POLICY_CACHE = OUTPUT_DIR / "asr_dual_selector_best_oof_policy_cache.jsonl"

SUMMARY_JSON = OUTPUT_DIR / "step31_dual_full_cache_summary.json"


# =========================================================
# 基础工具
# =========================================================

def normalize_path(path):

    text = str(path).strip().replace("\\", "/").lower()

    marker = "/data/testa/"

    if marker in text:
        return text.split(marker, 1)[1]

    return text


def remove_symbol_other(text):
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) != "So"
    )


def normalize_text(text, sensevoice=False):
    if text is None:
        return ""

    text = str(text).strip()

    if sensevoice:
        text = remove_symbol_other(text)

    text = re.sub(r"\s+", "", text)

    text = re.sub(
        r"[，。！？、；：“”‘’（）()【】\[\],.!?;:'\"—\-]",
        "",
        text,
    )

    return text


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_cache(path, sensevoice=False):
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
                raw = item.get("command_path")

                if not raw:
                    continue

                key = normalize_path(raw)

            cache[key] = {
                "command_key": key,
                "command_path": item.get("command_path", ""),
                "text": normalize_text(
                    item.get("text", ""),
                    sensevoice=sensevoice,
                ),
                "elapsed": float(
                    item.get("elapsed", 0.0) or 0.0
                ),
            }

    return cache


def write_cache(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# =========================================================
# 双模型selector特征（必须与Step30一致）
# =========================================================

def chinese_ratio(text):
    if not text:
        return 0.0
    n = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    return n / len(text)


def ascii_ratio(text):
    if not text:
        return 0.0
    n = sum(ord(ch) < 128 for ch in text)
    return n / len(text)


def digit_ratio(text):
    if not text:
        return 0.0
    n = sum(ch.isdigit() for ch in text)
    return n / len(text)


def repeated_ratio(text):
    if len(text) < 2:
        return 0.0
    repeats = sum(
        text[i] == text[i - 1]
        for i in range(1, len(text))
    )
    return repeats / (len(text) - 1)


def lcs_like_overlap(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    sa = set(a)
    sb = set(b)
    union = len(sa | sb)

    return len(sa & sb) / union if union else 0.0


def common_prefix_ratio(a, b):
    if not a or not b:
        return 0.0

    n = 0

    for x, y in zip(a, b):
        if x != y:
            break
        n += 1

    return n / max(len(a), len(b), 1)


def common_suffix_ratio(a, b):
    if not a or not b:
        return 0.0

    n = 0

    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            break
        n += 1

    return n / max(len(a), len(b), 1)


DOMAIN_WORDS = [
    "空调", "制冷", "制热", "除湿", "抽湿", "新风", "送风",
    "自动模式", "睡眠模式", "节能模式", "防直吹",
    "上下扫风", "左右扫风", "扫风",
    "风速", "风量", "风向", "出风口",
    "温度", "控温", "显示屏",
    "灯光", "亮度", "色温", "冷色调", "暖色调",
    "窗帘", "客厅", "厨房", "油烟机", "烟机", "滤网",
    "调高", "调低", "调大", "调小",
    "十六度", "十七度", "十八度", "十九度",
    "二十度", "二十一度", "二十二度", "二十三度",
    "二十四度", "二十五度", "二十六度", "二十七度",
    "二十八度", "二十九度", "三十度",
]


def domain_hits(text):
    return sum(word in text for word in DOMAIN_WORDS)


def make_numeric_features(para, sv, speaker_score):
    lp = len(para)
    ls = len(sv)

    inter_dist = editdistance.eval(para, sv)
    inter_norm = inter_dist / max(lp, ls, 1)

    return [
        lp,
        ls,
        lp - ls,
        abs(lp - ls),
        inter_dist,
        inter_norm,
        1.0 if para == sv else 0.0,
        1.0 if not para else 0.0,
        1.0 if not sv else 0.0,
        chinese_ratio(para),
        chinese_ratio(sv),
        ascii_ratio(para),
        ascii_ratio(sv),
        digit_ratio(para),
        digit_ratio(sv),
        repeated_ratio(para),
        repeated_ratio(sv),
        lcs_like_overlap(para, sv),
        common_prefix_ratio(para, sv),
        common_suffix_ratio(para, sv),
        domain_hits(para),
        domain_hits(sv),
        domain_hits(para) - domain_hits(sv),
        speaker_score,
    ]


# =========================================================
# 检查
# =========================================================

for required in [
    POS_JSONL,
    NEG_JSONL,
    PARA_CACHE_PATH,
    SELECTOR_MODEL,
    STEP30_OOF,
]:
    if not required.exists():
        raise FileNotFoundError(f"缺少文件：{required}")


# =========================================================
# 数据
# =========================================================

pos_items = read_jsonl(POS_JSONL)
neg_items = read_jsonl(NEG_JSONL)

samples = []

for split, items in [
    ("pos", pos_items),
    ("neg", neg_items),
]:
    for item in items:
        audio_path = DATA_DIR / item["识别音频"]
        key = normalize_path(audio_path)

        samples.append(
            {
                "split": split,
                "command_key": key,
                "command_path": audio_path,
                "reference": (
                    normalize_text(item.get("识别文本", ""))
                    if split == "pos"
                    else ""
                ),
            }
        )

print("正样本：", len(pos_items))
print("负样本：", len(neg_items))
print("总样本：", len(samples))


# =========================================================
# Paraformer
# =========================================================

para_cache = load_cache(
    PARA_CACHE_PATH,
    sensevoice=False,
)

missing_para = [
    s for s in samples
    if s["command_key"] not in para_cache
]

if missing_para:
    raise RuntimeError(
        f"Paraformer缓存缺少{len(missing_para)}条"
    )

print("Paraformer缓存：", len(para_cache))


# =========================================================
# SenseVoice：合并Step29正样本缓存 + 已有full缓存
# =========================================================

sv_cache = {}

step29_cache = load_cache(
    STEP29_SV_CACHE,
    sensevoice=True,
)

sv_cache.update(step29_cache)

full_existing = load_cache(
    FULL_SV_CACHE,
    sensevoice=True,
)

sv_cache.update(full_existing)

existing_target_count = sum(
    s["command_key"] in sv_cache
    for s in samples
)

missing_sv = [
    s for s in samples
    if s["command_key"] not in sv_cache
]

print("SenseVoice已存在目标样本：", existing_target_count)
print("SenseVoice本次需补：", len(missing_sv))

if missing_sv:

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("设备：", device)
    print("加载SenseVoiceSmall……")

    model = AutoModel(
        model="iic/SenseVoiceSmall",
        device=device,
        disable_update=True,
    )

    print("SenseVoiceSmall加载完成")

    for sample in tqdm(
        missing_sv,
        desc="Step31 SenseVoice补齐全量",
    ):
        start = time.perf_counter()

        with torch.inference_mode():
            result = model.generate(
                input=str(sample["command_path"]),
                cache={},
                language="zh",
                use_itn=False,
            )

        elapsed = time.perf_counter() - start

        text = ""

        if (
            result
            and isinstance(result, list)
            and isinstance(result[0], dict)
        ):
            raw = result[0].get("text", "")

            try:
                raw = rich_transcription_postprocess(raw)
            except Exception:
                pass

            text = normalize_text(
                raw,
                sensevoice=True,
            )

        sv_cache[sample["command_key"]] = {
            "command_key": sample["command_key"],
            "command_path": str(sample["command_path"]),
            "text": text,
            "elapsed": elapsed,
        }


# 写 canonical full cache
sv_full_records = []

for sample in samples:
    item = sv_cache[sample["command_key"]]

    sv_full_records.append(
        {
            "command_key": sample["command_key"],
            "command_path": str(sample["command_path"]),
            "text": item["text"],
            "elapsed": item.get("elapsed", 0.0),
            "split": sample["split"],
        }
    )

write_cache(
    FULL_SV_CACHE,
    sv_full_records,
)

print("SenseVoice全量缓存已写入：", FULL_SV_CACHE)


# =========================================================
# speaker score
# =========================================================

speaker_scores = {}

if STEP18_DETAILS.exists():
    with STEP18_DETAILS.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            raw = row.get("command_path")
            score = row.get("speaker_score")

            if raw and score not in ("", None):
                speaker_scores[
                    normalize_path(raw)
                ] = float(score)

matched_scores = sum(
    s["command_key"] in speaker_scores
    for s in samples
)

print("speaker_score匹配：", matched_scores, "/", len(samples))


# =========================================================
# 最终selector概率
# =========================================================

bundle = joblib.load(SELECTOR_MODEL)

vec_para = bundle["vectorizer_para"]
vec_sv = bundle["vectorizer_sensevoice"]
classifier = bundle["classifier"]

best_threshold = float(
    bundle.get("threshold", 0.5)
)

para_texts = [
    para_cache[s["command_key"]]["text"]
    for s in samples
]

sv_texts = [
    sv_cache[s["command_key"]]["text"]
    for s in samples
]

xp = vec_para.transform(para_texts)
xs = vec_sv.transform(sv_texts)

xn = csr_matrix(
    np.asarray(
        [
            make_numeric_features(
                para_texts[i],
                sv_texts[i],
                speaker_scores.get(
                    samples[i]["command_key"],
                    0.0,
                ),
            )
            for i in range(len(samples))
        ],
        dtype=np.float32,
    )
)

x = hstack(
    [xp, xs, xn],
    format="csr",
)

selector_probs = classifier.predict_proba(x)[:, 1]

print("最终selector阈值：", best_threshold)


# =========================================================
# Step30 OOF概率
# =========================================================

oof_prob_map = {}

with STEP30_OOF.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:
    reader = csv.DictReader(f)

    for row in reader:
        raw = row["command_path"]

        oof_prob_map[
            normalize_path(raw)
        ] = float(
            row["sensevoice_probability"]
        )

print("Step30 OOF正样本概率：", len(oof_prob_map))


# =========================================================
# 构建4套缓存
# =========================================================

shorter_records = []
t050_records = []
best_records = []
oof_policy_records = []

# 正样本评价
stats = {
    "para": {
        "distance": 0,
        "chars": 0,
    },
    "sensevoice": {
        "distance": 0,
        "chars": 0,
    },
    "shorter": {
        "distance": 0,
        "chars": 0,
    },
    "selector_t050_full_model": {
        "distance": 0,
        "chars": 0,
    },
    "selector_best_full_model": {
        "distance": 0,
        "chars": 0,
    },
    "selector_best_oof": {
        "distance": 0,
        "chars": 0,
    },
}

for i, sample in enumerate(samples):

    key = sample["command_key"]

    para = para_texts[i]
    sv = sv_texts[i]

    prob = float(selector_probs[i])

    # 1. 更短文本规则
    # 长度相同则优先SenseVoice（单模型CER更低）
    if len(sv) <= len(para):
        shorter_text = sv
        shorter_model = "sensevoice"
    else:
        shorter_text = para
        shorter_model = "paraformer"

    # 2. t=0.50
    if prob >= 0.50:
        t050_text = sv
        t050_model = "sensevoice"
    else:
        t050_text = para
        t050_model = "paraformer"

    # 3. Step30最佳阈值
    if prob >= best_threshold:
        best_text = sv
        best_model = "sensevoice"
    else:
        best_text = para
        best_model = "paraformer"

    # 4. OOF-policy cache
    if sample["split"] == "pos":
        if key not in oof_prob_map:
            raise RuntimeError(
                f"正样本缺少OOF selector概率："
                f"{sample['command_path']}"
            )

        oof_prob = oof_prob_map[key]

        if oof_prob >= best_threshold:
            oof_text = sv
            oof_model = "sensevoice"
        else:
            oof_text = para
            oof_model = "paraformer"

    else:
        # selector从未用负样本标签训练，
        # 对负样本使用最终模型不会造成正样本CER泄漏
        oof_prob = prob
        oof_text = best_text
        oof_model = best_model

    common = {
        "command_key": key,
        "command_path": str(sample["command_path"]),
        "split": sample["split"],
        "paraformer_text": para,
        "sensevoice_text": sv,
    }

    shorter_records.append(
        {
            **common,
            "text": shorter_text,
            "selected_model": shorter_model,
        }
    )

    t050_records.append(
        {
            **common,
            "text": t050_text,
            "selected_model": t050_model,
            "sensevoice_probability": prob,
            "selector_threshold": 0.50,
        }
    )

    best_records.append(
        {
            **common,
            "text": best_text,
            "selected_model": best_model,
            "sensevoice_probability": prob,
            "selector_threshold": best_threshold,
        }
    )

    oof_policy_records.append(
        {
            **common,
            "text": oof_text,
            "selected_model": oof_model,
            "sensevoice_probability": oof_prob,
            "selector_threshold": best_threshold,
            "selection_probability_source": (
                "oof"
                if sample["split"] == "pos"
                else "final_model"
            ),
        }
    )

    if sample["split"] == "pos":

        ref = sample["reference"]
        chars = len(ref)

        candidates = {
            "para": para,
            "sensevoice": sv,
            "shorter": shorter_text,
            "selector_t050_full_model": t050_text,
            "selector_best_full_model": best_text,
            "selector_best_oof": oof_text,
        }

        for name, hyp in candidates.items():
            stats[name]["chars"] += chars
            stats[name]["distance"] += editdistance.eval(
                ref,
                hyp,
            )


write_cache(SHORTER_CACHE, shorter_records)
write_cache(T050_CACHE, t050_records)
write_cache(BEST_CACHE, best_records)
write_cache(OOF_POLICY_CACHE, oof_policy_records)


# =========================================================
# 输出
# =========================================================

summary = {
    "positive_count": len(pos_items),
    "negative_count": len(neg_items),
    "total_count": len(samples),
    "selector_best_threshold": best_threshold,
    "positive_cer": {},
    "files": {
        "sensevoice_full": str(FULL_SV_CACHE),
        "dual_shorter": str(SHORTER_CACHE),
        "dual_selector_t050": str(T050_CACHE),
        "dual_selector_best": str(BEST_CACHE),
        "dual_selector_best_oof_policy": str(
            OOF_POLICY_CACHE
        ),
    },
}

print()
print("=" * 88)
print("Step31 正样本CER汇总")
print("=" * 88)

for name, item in stats.items():

    cer = (
        item["distance"] / item["chars"]
        if item["chars"]
        else 0.0
    )

    summary["positive_cer"][name] = cer

    suffix = ""

    if name in {
        "selector_t050_full_model",
        "selector_best_full_model",
    }:
        suffix = "  [训练集回看，仅供参考]"

    if name == "selector_best_oof":
        suffix = "  [严格OOF，可信]"

    print(
        f"{name:<28} "
        f"CER={cer * 100:.3f}%"
        f"{suffix}"
    )


# 选择率
def select_rate(records, model_name):
    return (
        sum(
            r.get("selected_model") == model_name
            for r in records
        )
        / len(records)
    )


summary["sensevoice_select_rate"] = {
    "shorter": select_rate(
        shorter_records,
        "sensevoice",
    ),
    "t050": select_rate(
        t050_records,
        "sensevoice",
    ),
    "best": select_rate(
        best_records,
        "sensevoice",
    ),
    "oof_policy_mixed": select_rate(
        oof_policy_records,
        "sensevoice",
    ),
}

print()
print("SenseVoice选择率（1838全量）：")

for name, rate in summary[
    "sensevoice_select_rate"
].items():
    print(
        f"{name:<20} {rate * 100:.2f}%"
    )


with SUMMARY_JSON.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("=" * 88)
print("缓存生成完成")
print("=" * 88)
print("SenseVoice全量：", FULL_SV_CACHE)
print("短文本规则：", SHORTER_CACHE)
print("Selector t=0.50：", T050_CACHE)
print("Selector最佳阈值：", BEST_CACHE)
print("OOF门控策略缓存：", OOF_POLICY_CACHE)
print("汇总：", SUMMARY_JSON)

