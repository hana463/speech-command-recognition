

import csv
import json
import re
from pathlib import Path

import editdistance
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


# =========================================================
# 双ASR + 文本门控 严格OOF联合优化
#
# 设计原则：
# 1. 正样本策略验证使用 Step30 OOF selector 产生的文本
# 2. 负样本 selector 从未使用负样本标签训练，因此可使用最终selector文本
# 3. 文本门控自身再做 StratifiedGroupKFold OOF
# 4. 门控参数只看 OOF 概率
# 5. 最终文本门控模型另用 full selector cache 全量训练
# =========================================================

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data" / "testA"
OUTPUT_DIR = PROJECT / "output"

POS_JSONL = DATA_DIR / "pos.jsonl"
NEG_JSONL = DATA_DIR / "neg.jsonl"

OOF_ASR_CACHE = (
    OUTPUT_DIR
    / "asr_dual_selector_best_oof_policy_cache.jsonl"
)

FULL_ASR_CACHE = (
    OUTPUT_DIR
    / "asr_dual_selector_best_full_cache.jsonl"
)

PARA_CACHE = (
    OUTPUT_DIR
    / "asr_hotword_best_full_cache.jsonl"
)

SV_CACHE = (
    OUTPUT_DIR
    / "asr_sensevoice_full_cache.jsonl"
)

SPEAKER_DETAIL_CSV = (
    OUTPUT_DIR
    / "final_testA_inference_details_optimized.csv"
)

MODEL_PATH = (
    OUTPUT_DIR
    / "dual_text_gate_model.joblib"
)

POLICY_PATH = (
    OUTPUT_DIR
    / "dual_text_gate_policy_oof.json"
)

OOF_DETAIL_CSV = (
    OUTPUT_DIR
    / "step32_dual_text_gate_oof_details.csv"
)

GRID_CSV = (
    OUTPUT_DIR
    / "step32_dual_text_gate_grid.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "step32_dual_text_gate_summary.json"
)

SHORTER_STRICT_CACHE = (
    OUTPUT_DIR
    / "asr_dual_shorter_strict_full_cache.jsonl"
)


# =========================================================
# 基础工具
# =========================================================

def normalize_path(path):

    text = str(path).strip().replace("\\", "/").lower()

    marker = "/data/testa/"

    if marker in text:
        return text.split(marker, 1)[1]

    return text


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", "", text)

    text = re.sub(
        r"[，。！？、；：“”‘’（）()【】\[\],.!?;:'\"—\-]",
        "",
        text,
    )

    return text


def model_text(text):
    text = normalize_text(text)
    return text if text else "空文本占位"


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

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            key = item.get("command_key")

            if key:
                key = normalize_path(key)
            else:
                raw = item.get("command_path")

                if not raw:
                    continue

                key = normalize_path(raw)

            cache[key] = {
                **item,
                "text": normalize_text(
                    item.get("text", "")
                ),
            }

    return cache


def predict_positive_probability(
    vectorizer,
    classifier,
    texts,
):
    x = vectorizer.transform(
        [model_text(t) for t in texts]
    )

    probs = classifier.predict_proba(x)

    classes = list(classifier.classes_)

    if 1 not in classes:
        raise RuntimeError(
            f"分类器classes_中没有正类1：{classes}"
        )

    pos_col = classes.index(1)

    return probs[:, pos_col]


# =========================================================
# 文件检查
# =========================================================

required_files = [
    POS_JSONL,
    NEG_JSONL,
    OOF_ASR_CACHE,
    FULL_ASR_CACHE,
    PARA_CACHE,
    SV_CACHE,
    SPEAKER_DETAIL_CSV,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"缺少文件：{path}"
        )


# =========================================================
# 标签
# =========================================================

labels = {}
references = {}
splits = {}
command_paths = {}

for item in read_jsonl(POS_JSONL):
    path = DATA_DIR / item["识别音频"]
    key = normalize_path(path)

    labels[key] = 1
    references[key] = normalize_text(
        item.get("识别文本", "")
    )
    splits[key] = "pos"
    command_paths[key] = str(path)

for item in read_jsonl(NEG_JSONL):
    path = DATA_DIR / item["识别音频"]
    key = normalize_path(path)

    labels[key] = 0
    references[key] = ""
    splits[key] = "neg"
    command_paths[key] = str(path)

all_keys = list(labels.keys())

print("标签总数：", len(all_keys))
print(
    "正样本：",
    sum(labels[k] == 1 for k in all_keys),
)
print(
    "负样本：",
    sum(labels[k] == 0 for k in all_keys),
)


# =========================================================
# speaker score
# =========================================================

speaker_scores = {}

with SPEAKER_DETAIL_CSV.open(
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

missing_scores = [
    k for k in all_keys
    if k not in speaker_scores
]

if missing_scores:
    raise RuntimeError(
        f"speaker_score缺少"
        f"{len(missing_scores)}条"
    )

print(
    "speaker_score：",
    len(speaker_scores),
)


# =========================================================
# ASR缓存
# =========================================================

oof_cache = load_cache(
    OOF_ASR_CACHE
)

full_cache = load_cache(
    FULL_ASR_CACHE
)

para_cache = load_cache(
    PARA_CACHE
)

sv_cache = load_cache(
    SV_CACHE
)

for name, cache in [
    ("OOF双ASR", oof_cache),
    ("Full双ASR", full_cache),
    ("Paraformer", para_cache),
    ("SenseVoice", sv_cache),
]:
    missing = [
        k for k in all_keys
        if k not in cache
    ]

    if missing:
        raise RuntimeError(
            f"{name}缺少{len(missing)}条"
        )

    print(
        f"{name}缓存：",
        len(cache),
    )


# =========================================================
# 修复 shorter 备用缓存
# =========================================================

shorter_records = []

for key in all_keys:
    para = para_cache[key]["text"]
    sv = sv_cache[key]["text"]

    # 与Step30完全一致：
    # 只有SenseVoice严格更短才选SenseVoice
    # 等长默认Paraformer
    if len(sv) < len(para):
        text = sv
        selected_model = "sensevoice"
    else:
        text = para
        selected_model = "paraformer"

    shorter_records.append(
        {
            "command_key": key,
            "command_path": command_paths[key],
            "split": splits[key],
            "text": text,
            "selected_model": selected_model,
            "paraformer_text": para,
            "sensevoice_text": sv,
        }
    )

with SHORTER_STRICT_CACHE.open(
    "w",
    encoding="utf-8",
) as f:
    for item in shorter_records:
        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )

shorter_sv_rate = (
    sum(
        r["selected_model"] == "sensevoice"
        for r in shorter_records
    )
    / len(shorter_records)
)

# 正样本CER检查
shorter_chars = 0
shorter_dist = 0

for item in shorter_records:
    if item["split"] != "pos":
        continue

    ref = references[
        item["command_key"]
    ]

    shorter_chars += len(ref)
    shorter_dist += editdistance.eval(
        ref,
        item["text"],
    )

shorter_cer = (
    shorter_dist / shorter_chars
    if shorter_chars
    else 0.0
)

print()
print(
    "修复后的shorter CER：",
    f"{shorter_cer * 100:.3f}%",
)
print(
    "修复后的shorter SenseVoice选择率：",
    f"{shorter_sv_rate * 100:.2f}%",
)


# =========================================================
# OOF文本门控数据
# =========================================================

rows = []

for key in all_keys:
    text = oof_cache[key]["text"]
    label = labels[key]
    reference = references[key]

    if label == 1:
        ref_len = len(reference)
        asr_distance = editdistance.eval(
            reference,
            text,
        )
    else:
        ref_len = 0
        asr_distance = 0

    rows.append(
        {
            "key": key,
            "command_path": command_paths[key],
            "split": splits[key],
            "label": label,
            "reference": reference,
            "reference_length": ref_len,
            "asr_text": text,
            "asr_distance": asr_distance,
            "asr_empty": text == "",
            "speaker_score": speaker_scores[key],
        }
    )

n = len(rows)

y = np.asarray(
    [r["label"] for r in rows],
    dtype=np.int32,
)

groups = np.asarray(
    [
        model_text(r["asr_text"])
        for r in rows
    ],
    dtype=object,
)

unique_groups = len(set(groups.tolist()))

print()
print("OOF门控样本：", n)
print("不同ASR文本组：", unique_groups)


# =========================================================
# 5折 OOF 文本概率
# =========================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=20260807,
)

oof_prob = np.full(
    n,
    np.nan,
    dtype=np.float64,
)

oof_fold = np.full(
    n,
    -1,
    dtype=np.int32,
)

print("开始生成双ASR文本门控 OOF 概率……")

for fold, (train_idx, val_idx) in enumerate(
    cv.split(
        np.zeros(n),
        y,
        groups=groups,
    ),
    start=1,
):
    train_texts = [
        rows[i]["asr_text"]
        for i in train_idx
    ]

    val_texts = [
        rows[i]["asr_text"]
        for i in val_idx
    ]

    y_train = y[train_idx]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 4),
        min_df=2,
        max_features=16000,
        sublinear_tf=True,
    )

    x_train = vectorizer.fit_transform(
        [
            model_text(t)
            for t in train_texts
        ]
    )

    classifier = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=20260807 + fold,
    )

    classifier.fit(
        x_train,
        y_train,
    )

    prob = predict_positive_probability(
        vectorizer,
        classifier,
        val_texts,
    )

    oof_prob[val_idx] = prob
    oof_fold[val_idx] = fold

    print(
        f"第{fold}折："
        f"train={len(train_idx)}，"
        f"val={len(val_idx)}"
    )

if np.isnan(oof_prob).any():
    raise RuntimeError(
        "仍有样本没有OOF文本概率"
    )

pred = (
    oof_prob >= 0.5
).astype(np.int32)

oof_acc = accuracy_score(
    y,
    pred,
)

try:
    oof_auc = roc_auc_score(
        y,
        oof_prob,
    )
except Exception:
    oof_auc = None

print()
print(
    "双ASR文本门控 OOF准确率：",
    oof_acc,
)
print(
    "双ASR文本门控 OOF ROC-AUC：",
    oof_auc,
)


# =========================================================
# 基础数组
# =========================================================

speaker_array = np.asarray(
    [r["speaker_score"] for r in rows],
    dtype=np.float64,
)

ref_lengths = np.asarray(
    [r["reference_length"] for r in rows],
    dtype=np.int32,
)

asr_distances = np.asarray(
    [r["asr_distance"] for r in rows],
    dtype=np.int32,
)

asr_empty = np.asarray(
    [r["asr_empty"] for r in rows],
    dtype=bool,
)

pos_mask = y == 1
neg_mask = y == 0

total_pos_chars = int(
    np.sum(
        ref_lengths[pos_mask]
    )
)

pure_oof_asr_distance = int(
    np.sum(
        asr_distances[pos_mask]
    )
)

pure_oof_asr_cer = (
    pure_oof_asr_distance
    / total_pos_chars
)

print()
print(
    "门控前严格OOF双ASR CER：",
    f"{pure_oof_asr_cer * 100:.3f}%",
)


# =========================================================
# 策略评估
# =========================================================

def evaluate_policy(
    low_threshold,
    high_threshold,
    p_threshold,
):
    low_region = (
        speaker_array
        < low_threshold
    )

    high_region = (
        speaker_array
        >= high_threshold
    )

    middle_region = (
        (~low_region)
        & (~high_region)
    )

    accepted = (
        high_region
        | (
            middle_region
            & (
                oof_prob
                >= p_threshold
            )
        )
    )

    dual_asr_called = ~low_region

    text_gate_called = middle_region

    # 正样本：拒绝 -> 空串
    final_pos_distance = np.where(
        accepted[pos_mask],
        asr_distances[pos_mask],
        ref_lengths[pos_mask],
    )

    total_distance = int(
        np.sum(
            final_pos_distance
        )
    )

    cer = (
        total_distance
        / total_pos_chars
    )

    # 负样本：
    # 门控拒绝 或 ASR本身空串
    neg_final_empty = (
        (~accepted[neg_mask])
        | asr_empty[neg_mask]
    )

    rr = float(
        np.mean(
            neg_final_empty
        )
    )

    call_rate = float(
        np.mean(
            dual_asr_called
        )
    )

    text_gate_rate = float(
        np.mean(
            text_gate_called
        )
    )

    recognition_score = (
        0.5 * (1.0 - cer)
        + 0.5 * rr
    )

    # 双ASR：每个进入ASR区的样本理论上会调用2个ASR
    equivalent_asr_calls = (
        2.0 * call_rate
    )

    # 不是官方效率分，只做本地参考
    efficiency_proxy = (
        1.0
        - min(
            1.0,
            equivalent_asr_calls / 2.0,
        )
    )

    contest_proxy = (
        0.4 * (1.0 - cer)
        + 0.4 * rr
        + 0.2 * efficiency_proxy
    )

    return {
        "low": float(low_threshold),
        "high": float(high_threshold),
        "p": float(p_threshold),
        "cer": float(cer),
        "rr": float(rr),
        "recognition_score": float(
            recognition_score
        ),
        "contest_proxy": float(
            contest_proxy
        ),
        "dual_asr_call_rate": call_rate,
        "equivalent_asr_calls_per_sample": (
            equivalent_asr_calls
        ),
        "text_gate_call_rate": (
            text_gate_rate
        ),
        "low_reject_count": int(
            np.sum(low_region)
        ),
        "high_accept_count": int(
            np.sum(high_region)
        ),
        "middle_accept_count": int(
            np.sum(
                middle_region
                & accepted
            )
        ),
        "middle_reject_count": int(
            np.sum(
                middle_region
                & (~accepted)
            )
        ),
        "total_edit_distance": (
            total_distance
        ),
    }


# =========================================================
# 网格
# =========================================================

LOW_VALUES = [
    round(x / 1000, 3)
    for x in range(-50, 101, 5)
]  # -0.050 ~ 0.100

HIGH_VALUES = [
    round(x / 1000, 3)
    for x in range(300, 461, 5)
]  # 0.300 ~ 0.460

P_VALUES = [
    round(x / 100, 2)
    for x in range(40, 91, 2)
]  # 0.40 ~ 0.90

grid = []

print()
print("开始搜索双ASR门控参数……")

for low in LOW_VALUES:
    for high in HIGH_VALUES:
        if low >= high:
            continue

        for p in P_VALUES:
            grid.append(
                evaluate_policy(
                    low,
                    high,
                    p,
                )
            )

print(
    "搜索策略数量：",
    len(grid),
)


# =========================================================
# 候选选择
# =========================================================

recognition_best = max(
    grid,
    key=lambda r: (
        r["recognition_score"],
        -r["cer"],
        r["rr"],
    ),
)

# 识别分距离最优<=0.0015时，优先更少双ASR调用
near_best = [
    r for r in grid
    if (
        recognition_best[
            "recognition_score"
        ]
        - r["recognition_score"]
        <= 0.0015
    )
]

hidden_b_robust = min(
    near_best,
    key=lambda r: (
        r["dual_asr_call_rate"],
        r["cer"],
        -r["rr"],
    ),
)

# 与之前效率策略保持一致：
# 双ASR整体调用率 <= 90%
efficient_pool = [
    r for r in grid
    if (
        r["dual_asr_call_rate"]
        <= 0.90
    )
]

efficient_90 = max(
    efficient_pool,
    key=lambda r: (
        r["recognition_score"],
        -r["cer"],
        r["rr"],
    ),
) if efficient_pool else None


RR_FLOORS = [
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
]

best_by_rr = {}

for floor in RR_FLOORS:
    candidates = [
        r for r in grid
        if r["rr"] >= floor
    ]

    if not candidates:
        continue

    best = min(
        candidates,
        key=lambda r: (
            r["cer"],
            -r["recognition_score"],
            r["dual_asr_call_rate"],
        ),
    )

    best_by_rr[str(floor)] = best


# =========================================================
# 固定策略对照
# =========================================================

COMPARE_PROFILES = {
    "old_competition_gate": (
        0.035,
        0.400,
        0.74,
    ),
    "old_safe_gate": (
        -0.020,
        0.380,
        0.62,
    ),
}

profile_results = {
    name: evaluate_policy(
        *params
    )
    for name, params
    in COMPARE_PROFILES.items()
}


# =========================================================
# 输出
# =========================================================

print()
print("=" * 92)
print("Step32 固定策略对比")
print("=" * 92)

for name, result in profile_results.items():
    print(
        f"{name:<24} "
        f"CER={result['cer']*100:7.3f}%  "
        f"RR={result['rr']*100:7.3f}%  "
        f"Rec={result['recognition_score']:.6f}  "
        f"DualASR={result['dual_asr_call_rate']*100:6.2f}%"
    )

print()
print("=" * 92)
print("Step32 推荐")
print("=" * 92)

print("OOF识别平衡最高：")
print(recognition_best)

print()
print("隐藏B稳健推荐：")
print(hidden_b_robust)

print()
print("效率友好 DualASR<=90%：")
print(efficient_90)

print()
print("不同RR下限最低CER：")

for floor, result in best_by_rr.items():
    print(
        f"RR>={float(floor)*100:.1f}%："
        f"CER={result['cer']*100:.3f}% "
        f"RR={result['rr']*100:.3f}% "
        f"low={result['low']:.3f} "
        f"high={result['high']:.3f} "
        f"p={result['p']:.2f} "
        f"DualASR={result['dual_asr_call_rate']*100:.2f}%"
    )


# =========================================================
# 保存OOF详情
# =========================================================

with OOF_DETAIL_CSV.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:
    fields = [
        "fold",
        "command_path",
        "split",
        "reference",
        "asr_text",
        "asr_distance",
        "speaker_score",
        "text_probability",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()

    for i, row in enumerate(rows):
        writer.writerow(
            {
                "fold": int(
                    oof_fold[i]
                ),
                "command_path": (
                    row[
                        "command_path"
                    ]
                ),
                "split": row["split"],
                "reference": (
                    row["reference"]
                ),
                "asr_text": (
                    row["asr_text"]
                ),
                "asr_distance": (
                    row[
                        "asr_distance"
                    ]
                ),
                "speaker_score": (
                    row[
                        "speaker_score"
                    ]
                ),
                "text_probability": (
                    float(
                        oof_prob[i]
                    )
                ),
            }
        )


# =========================================================
# 保存网格
# =========================================================

with GRID_CSV.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            grid[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        sorted(
            grid,
            key=lambda r: (
                -r[
                    "recognition_score"
                ],
                r["cer"],
            ),
        )
    )


# =========================================================
# 最终文本门控模型
# 用 full selector cache 训练
# =========================================================

full_texts = [
    full_cache[k]["text"]
    for k in all_keys
]

full_y = np.asarray(
    [
        labels[k]
        for k in all_keys
    ],
    dtype=np.int32,
)

final_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(1, 4),
    min_df=2,
    max_features=16000,
    sublinear_tf=True,
)

x_full = final_vectorizer.fit_transform(
    [
        model_text(t)
        for t in full_texts
    ]
)

final_classifier = LogisticRegression(
    C=2.0,
    class_weight="balanced",
    max_iter=3000,
    solver="liblinear",
    random_state=20260807,
)

final_classifier.fit(
    x_full,
    full_y,
)


# =========================================================
# 最终策略
#
# 默认写 hidden_b_robust
# 但同时保留 recognition / efficient 候选
# =========================================================

final_policy = hidden_b_robust

model_bundle = {
    "vectorizer": (
        final_vectorizer
    ),
    "classifier": (
        final_classifier
    ),
    "text_normalization": (
        "step32_normalize_text_v1"
    ),
    "training_cache": str(
        FULL_ASR_CACHE
    ),
    "selector_cache_type": (
        "full_model"
    ),
    "oof_validation": {
        "accuracy": (
            float(oof_acc)
        ),
        "roc_auc": (
            None
            if oof_auc is None
            else float(oof_auc)
        ),
        "pure_dual_asr_oof_cer": (
            float(
                pure_oof_asr_cer
            )
        ),
    },
}

joblib.dump(
    model_bundle,
    MODEL_PATH,
)

policy_payload = {
    "mode": (
        "dual_asr_text_gate"
    ),
    "low_threshold": (
        final_policy["low"]
    ),
    "high_threshold": (
        final_policy["high"]
    ),
    "text_probability_threshold": (
        final_policy["p"]
    ),
    "selector_threshold": 0.32,

    "oof_cer": (
        final_policy["cer"]
    ),
    "oof_rr": (
        final_policy["rr"]
    ),
    "oof_recognition_score": (
        final_policy[
            "recognition_score"
        ]
    ),
    "dual_asr_call_rate": (
        final_policy[
            "dual_asr_call_rate"
        ]
    ),
    "text_gate_call_rate": (
        final_policy[
            "text_gate_call_rate"
        ]
    ),

    "oof_text_gate_accuracy": (
        float(oof_acc)
    ),
    "oof_text_gate_auc": (
        None
        if oof_auc is None
        else float(oof_auc)
    ),

    "pure_dual_asr_oof_cer": (
        float(
            pure_oof_asr_cer
        )
    ),

    "alternatives": {
        "recognition_best": (
            recognition_best
        ),
        "hidden_b_robust": (
            hidden_b_robust
        ),
        "efficient_90": (
            efficient_90
        ),
        "best_by_rr_floor": (
            best_by_rr
        ),
    },

    "notes": [
        (
            "策略搜索使用OOF selector文本"
        ),
        (
            "文本门控概率也使用5折OOF"
        ),
        (
            "最终文本门控模型使用full selector缓存全量训练"
        ),
        (
            "equivalent_asr_calls_per_sample仅为本地效率参考，不是官方效率分"
        ),
    ],
}

with POLICY_PATH.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        policy_payload,
        f,
        ensure_ascii=False,
        indent=2,
    )


# =========================================================
# Summary
# =========================================================

summary = {
    "sample_count": n,
    "positive_count": int(
        np.sum(pos_mask)
    ),
    "negative_count": int(
        np.sum(neg_mask)
    ),

    "pure_dual_asr_oof_cer": (
        pure_oof_asr_cer
    ),

    "text_gate_oof_accuracy": (
        float(oof_acc)
    ),
    "text_gate_oof_auc": (
        None
        if oof_auc is None
        else float(oof_auc)
    ),

    "shorter_strict_cer": (
        shorter_cer
    ),
    "shorter_strict_sv_rate": (
        shorter_sv_rate
    ),

    "recognition_best": (
        recognition_best
    ),
    "hidden_b_robust": (
        hidden_b_robust
    ),
    "efficient_90": (
        efficient_90
    ),
    "best_by_rr_floor": (
        best_by_rr
    ),

    "files": {
        "model": str(
            MODEL_PATH
        ),
        "policy": str(
            POLICY_PATH
        ),
        "oof_details": str(
            OOF_DETAIL_CSV
        ),
        "grid": str(
            GRID_CSV
        ),
        "shorter_strict_cache": str(
            SHORTER_STRICT_CACHE
        ),
    },
}

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
print("=" * 92)
print("Step32 保存完成")
print("=" * 92)
print("最终文本门控模型：", MODEL_PATH)
print("OOF策略：", POLICY_PATH)
print("OOF详情：", OOF_DETAIL_CSV)
print("参数网格：", GRID_CSV)
print("修复shorter缓存：", SHORTER_STRICT_CACHE)
print("汇总：", SUMMARY_JSON)
