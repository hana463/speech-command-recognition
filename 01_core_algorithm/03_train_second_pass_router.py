

import csv
import json
import re
from pathlib import Path

import editdistance
import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


# =========================================================
# 选择性调用第二ASR（SenseVoice）OOF路由器
#
# 推理结构：
# Paraformer
#    ↓
# second-pass router
#    ├─ 不调用SV -> 直接保留Paraformer
#    └─ 调用SV   -> SenseVoice + Step30双ASR selector
#

# =========================================================

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT / "output"

STEP29_DETAILS = (
    OUTPUT_DIR
    / "step29_sensevoice_full_details.jsonl"
)

STEP30_OOF = (
    OUTPUT_DIR
    / "step30_dual_asr_selector_oof.csv"
)

STEP18_DETAILS = (
    OUTPUT_DIR
    / "final_testA_inference_details_optimized.csv"
)

ROUTER_MODEL = (
    OUTPUT_DIR
    / "dual_asr_second_pass_router.joblib"
)

REPORT_JSON = (
    OUTPUT_DIR
    / "step33_second_pass_router_validation.json"
)

OOF_CSV = (
    OUTPUT_DIR
    / "step33_second_pass_router_oof.csv"
)


# =========================================================
# 工具
# =========================================================

def normalize_path(path):
    return str(Path(path).resolve()).lower()


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", "", text)

    return text


def chinese_ratio(text):
    if not text:
        return 0.0

    return (
        sum(
            "\u4e00" <= ch <= "\u9fff"
            for ch in text
        )
        / len(text)
    )


def ascii_ratio(text):
    if not text:
        return 0.0

    return (
        sum(
            ord(ch) < 128
            for ch in text
        )
        / len(text)
    )


def digit_ratio(text):
    if not text:
        return 0.0

    return (
        sum(
            ch.isdigit()
            for ch in text
        )
        / len(text)
    )


def repeated_ratio(text):
    if len(text) < 2:
        return 0.0

    repeats = sum(
        text[i] == text[i - 1]
        for i in range(1, len(text))
    )

    return repeats / (len(text) - 1)


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
    return sum(
        word in text
        for word in DOMAIN_WORDS
    )


def numeric_features(text, speaker_score):
    return [
        len(text),
        chinese_ratio(text),
        ascii_ratio(text),
        digit_ratio(text),
        repeated_ratio(text),
        domain_hits(text),
        1.0 if not text else 0.0,
        speaker_score,
    ]


# =========================================================
# 检查
# =========================================================

for p in [
    STEP29_DETAILS,
    STEP30_OOF,
    STEP18_DETAILS,
]:
    if not p.exists():
        raise FileNotFoundError(
            f"缺少文件：{p}"
        )


# =========================================================
# speaker score
# =========================================================

speaker_scores = {}

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

print(
    "speaker_score：",
    len(speaker_scores),
)


# =========================================================
# Step30 OOF selector probability
# =========================================================

selector_oof = {}

with STEP30_OOF.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:
    reader = csv.DictReader(f)

    for row in reader:
        key = normalize_path(
            row["command_path"]
        )

        selector_oof[key] = float(
            row[
                "sensevoice_probability"
            ]
        )

print(
    "Step30 OOF概率：",
    len(selector_oof),
)


# =========================================================
# 正样本详情
# =========================================================

rows = []

with STEP29_DETAILS.open(
    "r",
    encoding="utf-8",
) as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        item = json.loads(line)

        path = item["command_path"]
        key = normalize_path(path)

        if key not in selector_oof:
            raise RuntimeError(
                f"缺少Step30 OOF概率：{path}"
            )

        if key not in speaker_scores:
            raise RuntimeError(
                f"缺少speaker score：{path}"
            )

        ref = normalize_text(
            item["reference"]
        )

        para = normalize_text(
            item["paraformer"]
        )

        sv = normalize_text(
            item["sensevoice"]
        )

        dp = int(
            item["para_distance"]
        )

        ds = int(
            item["sensevoice_distance"]
        )

        selector_prob = selector_oof[key]

        # Step30 OOF最佳threshold = 0.32
        if selector_prob >= 0.32:
            dual_selected_distance = ds
            dual_selected_model = "sensevoice"
        else:
            dual_selected_distance = dp
            dual_selected_model = "paraformer"

        gain = (
            dp
            - dual_selected_distance
        )

        # 真正值得调用第二ASR：
        # OOF dual selector最终结果比Paraformer更好
        beneficial = int(
            gain > 0
        )

        rows.append(
            {
                "key": key,
                "command_path": path,
                "reference": ref,
                "paraformer": para,
                "sensevoice": sv,
                "para_distance": dp,
                "sensevoice_distance": ds,
                "selector_oof_probability": (
                    selector_prob
                ),
                "dual_selected_model": (
                    dual_selected_model
                ),
                "dual_selected_distance": (
                    dual_selected_distance
                ),
                "gain_chars": gain,
                "beneficial": beneficial,
                "speaker_score": (
                    speaker_scores[key]
                ),
            }
        )


n = len(rows)

print("正样本：", n)
print(
    "调用SenseVoice后真正有收益：",
    sum(
        r["beneficial"]
        for r in rows
    ),
)

print(
    "无收益/有害：",
    sum(
        not r["beneficial"]
        for r in rows
    ),
)


# =========================================================
# OOF Router
# =========================================================

texts = [
    r["paraformer"]
    for r in rows
]

y = np.asarray(
    [
        r["beneficial"]
        for r in rows
    ],
    dtype=np.int32,
)

groups = np.asarray(
    [
        r["reference"]
        if r["reference"]
        else f"EMPTY_{i}"
        for i, r in enumerate(rows)
    ],
    dtype=object,
)

indices = np.arange(n)

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

fold_ids = np.full(
    n,
    -1,
    dtype=np.int32,
)


for fold, (train_idx, val_idx) in enumerate(
    cv.split(
        indices,
        y,
        groups=groups,
    ),
    start=1,
):

    train_texts = [
        texts[i]
        for i in train_idx
    ]

    val_texts = [
        texts[i]
        for i in val_idx
    ]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 4),
        min_df=2,
        max_features=14000,
        sublinear_tf=True,
    )

    xt_train = vectorizer.fit_transform(
        train_texts
    )

    xt_val = vectorizer.transform(
        val_texts
    )

    xn_train = csr_matrix(
        np.asarray(
            [
                numeric_features(
                    rows[i]["paraformer"],
                    rows[i]["speaker_score"],
                )
                for i in train_idx
            ],
            dtype=np.float32,
        )
    )

    xn_val = csr_matrix(
        np.asarray(
            [
                numeric_features(
                    rows[i]["paraformer"],
                    rows[i]["speaker_score"],
                )
                for i in val_idx
            ],
            dtype=np.float32,
        )
    )

    x_train = hstack(
        [
            xt_train,
            xn_train,
        ],
        format="csr",
    )

    x_val = hstack(
        [
            xt_val,
            xn_val,
        ],
        format="csr",
    )

    classifier = LogisticRegression(
        C=1.5,
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=20260807 + fold,
    )

    classifier.fit(
        x_train,
        y[train_idx],
    )

    classes = list(
        classifier.classes_
    )

    pos_col = classes.index(1)

    oof_prob[val_idx] = (
        classifier.predict_proba(
            x_val
        )[:, pos_col]
    )

    fold_ids[val_idx] = fold

    print(
        f"第{fold}折："
        f"train={len(train_idx)} "
        f"val={len(val_idx)}"
    )


if np.isnan(oof_prob).any():
    raise RuntimeError(
        "Router OOF概率不完整"
    )


# =========================================================
# 分类能力
# =========================================================

pred_05 = (
    oof_prob >= 0.5
).astype(np.int32)

acc = accuracy_score(
    y,
    pred_05,
)

try:
    auc = roc_auc_score(
        y,
        oof_prob,
    )
except Exception:
    auc = None


print()
print("=" * 88)
print("Step33 Router OOF分类能力")
print("=" * 88)
print(
    "OOF Accuracy：",
    acc,
)
print(
    "OOF ROC-AUC：",
    auc,
)


# =========================================================
# CER / 第二模型调用率
# =========================================================

total_chars = sum(
    len(r["reference"])
    for r in rows
)

para_dist = sum(
    r["para_distance"]
    for r in rows
)

full_dual_dist = sum(
    r["dual_selected_distance"]
    for r in rows
)

para_cer = (
    para_dist
    / total_chars
)

full_dual_cer = (
    full_dual_dist
    / total_chars
)


def evaluate_threshold(threshold):
    total_dist = 0
    second_calls = 0

    gain_recovered = 0
    harmful_calls = 0

    for i, row in enumerate(rows):

        call_sv = (
            oof_prob[i]
            >= threshold
        )

        if call_sv:
            second_calls += 1

            selected_dist = (
                row[
                    "dual_selected_distance"
                ]
            )

            total_dist += selected_dist

            gain = (
                row["para_distance"]
                - selected_dist
            )

            gain_recovered += gain

            if gain < 0:
                harmful_calls += 1

        else:
            total_dist += (
                row["para_distance"]
            )

    cer = (
        total_dist
        / total_chars
    )

    full_gain = (
        para_dist
        - full_dual_dist
    )

    gain_capture = (
        gain_recovered
        / full_gain
        if full_gain > 0
        else 0.0
    )

    return {
        "threshold": float(
            threshold
        ),
        "cer": float(cer),
        "second_asr_call_rate": (
            second_calls / n
        ),
        "sensevoice_calls": int(
            second_calls
        ),
        "gain_capture_ratio": float(
            gain_capture
        ),
        "harmful_second_calls": int(
            harmful_calls
        ),
        "total_edit_distance": int(
            total_dist
        ),
    }


results = []

for t in np.arange(
    0.05,
    0.951,
    0.01,
):
    results.append(
        evaluate_threshold(
            round(
                float(t),
                2,
            )
        )
    )


# =========================================================
# Pareto候选
# =========================================================

# 追求CER：第二调用<=70%
pools = {}

for max_rate in [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]:
    candidates = [
        r for r in results
        if (
            r[
                "second_asr_call_rate"
            ]
            <= max_rate
        )
    ]

    if candidates:
        pools[str(max_rate)] = min(
            candidates,
            key=lambda r: (
                r["cer"],
                r[
                    "second_asr_call_rate"
                ],
            ),
        )


# 自动稳健候选：
# 捕获>=80% full-dual收益时，尽量少调用第二ASR
capture_80 = [
    r for r in results
    if (
        r[
            "gain_capture_ratio"
        ]
        >= 0.80
    )
]

robust_80 = min(
    capture_80,
    key=lambda r: (
        r[
            "second_asr_call_rate"
        ],
        r["cer"],
    ),
) if capture_80 else None


# 捕获>=90%
capture_90 = [
    r for r in results
    if (
        r[
            "gain_capture_ratio"
        ]
        >= 0.90
    )
]

robust_90 = min(
    capture_90,
    key=lambda r: (
        r[
            "second_asr_call_rate"
        ],
        r["cer"],
    ),
) if capture_90 else None


# 最低CER
best_cer = min(
    results,
    key=lambda r: (
        r["cer"],
        r[
            "second_asr_call_rate"
        ],
    ),
)


# =========================================================
# 输出
# =========================================================

print()
print("=" * 88)
print("Step33 选择性双ASR OOF结果")
print("=" * 88)

print(
    f"Paraformer CER："
    f"{para_cer*100:.3f}%"
)

print(
    f"全量双ASR OOF CER："
    f"{full_dual_cer*100:.3f}%"
)

print()
print(
    "Router最低CER：",
    best_cer,
)

print()
print(
    "捕获>=80%融合收益的最省调用方案："
)
print(
    robust_80
)

print()
print(
    "捕获>=90%融合收益的最省调用方案："
)
print(
    robust_90
)

print()
print(
    "不同SenseVoice最大调用率："
)

for rate, result in pools.items():
    print(
        f"SV<={float(rate)*100:.0f}%："
        f"CER={result['cer']*100:.3f}% "
        f"threshold={result['threshold']:.2f} "
        f"SV调用={result['second_asr_call_rate']*100:.2f}% "
        f"收益捕获={result['gain_capture_ratio']*100:.1f}%"
    )


# =========================================================
# 保存OOF
# =========================================================

with OOF_CSV.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    fields = [
        "fold",
        "command_path",
        "reference",
        "paraformer",
        "sensevoice",
        "para_distance",
        "sensevoice_distance",
        "dual_selected_distance",
        "gain_chars",
        "beneficial",
        "speaker_score",
        "router_probability",
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
                    fold_ids[i]
                ),
                "command_path": (
                    row[
                        "command_path"
                    ]
                ),
                "reference": (
                    row["reference"]
                ),
                "paraformer": (
                    row["paraformer"]
                ),
                "sensevoice": (
                    row["sensevoice"]
                ),
                "para_distance": (
                    row[
                        "para_distance"
                    ]
                ),
                "sensevoice_distance": (
                    row[
                        "sensevoice_distance"
                    ]
                ),
                "dual_selected_distance": (
                    row[
                        "dual_selected_distance"
                    ]
                ),
                "gain_chars": (
                    row[
                        "gain_chars"
                    ]
                ),
                "beneficial": (
                    row["beneficial"]
                ),
                "speaker_score": (
                    row[
                        "speaker_score"
                    ]
                ),
                "router_probability": (
                    float(
                        oof_prob[i]
                    )
                ),
            }
        )


# =========================================================
# 最终Router模型
# =========================================================

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(1, 4),
    min_df=2,
    max_features=14000,
    sublinear_tf=True,
)

xt = vectorizer.fit_transform(
    texts
)

xn = csr_matrix(
    np.asarray(
        [
            numeric_features(
                r["paraformer"],
                r["speaker_score"],
            )
            for r in rows
        ],
        dtype=np.float32,
    )
)

x = hstack(
    [xt, xn],
    format="csr",
)

classifier = LogisticRegression(
    C=1.5,
    class_weight="balanced",
    max_iter=3000,
    solver="liblinear",
    random_state=20260807,
)

classifier.fit(
    x,
    y,
)

# 默认选择80%收益候选；
# 若不存在，则用最低CER候选
recommended = (
    robust_80
    if robust_80 is not None
    else best_cer
)

bundle = {
    "vectorizer": vectorizer,
    "classifier": classifier,
    "recommended_threshold": (
        recommended[
            "threshold"
        ]
    ),
    "selector_threshold": 0.32,
    "feature_version": 1,
    "numeric_feature_names": [
        "length",
        "chinese_ratio",
        "ascii_ratio",
        "digit_ratio",
        "repeated_ratio",
        "domain_hits",
        "is_empty",
        "speaker_score",
    ],
    "oof_metrics": {
        "accuracy_0_5": float(
            acc
        ),
        "roc_auc": (
            None
            if auc is None
            else float(auc)
        ),
        "paraformer_cer": float(
            para_cer
        ),
        "full_dual_oof_cer": float(
            full_dual_cer
        ),
        "best_cer": best_cer,
        "robust_80": robust_80,
        "robust_90": robust_90,
        "call_rate_pareto": pools,
    },
}

joblib.dump(
    bundle,
    ROUTER_MODEL,
)


report = {
    "sample_count": n,
    "beneficial_count": int(
        np.sum(y)
    ),
    "router_oof_accuracy_0_5": float(
        acc
    ),
    "router_oof_auc": (
        None
        if auc is None
        else float(auc)
    ),
    "paraformer_cer": float(
        para_cer
    ),
    "full_dual_oof_cer": float(
        full_dual_cer
    ),
    "best_cer": best_cer,
    "robust_80": robust_80,
    "robust_90": robust_90,
    "call_rate_pareto": pools,
}

with REPORT_JSON.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        report,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("=" * 88)
print("Step33 保存完成")
print("=" * 88)
print(
    "Router模型：",
    ROUTER_MODEL,
)
print(
    "报告：",
    REPORT_JSON,
)
print(
    "OOF详情：",
    OOF_CSV,
)
