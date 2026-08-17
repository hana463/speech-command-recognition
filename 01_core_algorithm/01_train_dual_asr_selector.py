

import csv
import json
import math
import re
from pathlib import Path

import editdistance
import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold


# =========================================================
# Step30：Paraformer / SenseVoice 双ASR OOF选择器
#
# 目标：
# - 只用“推理时可获得”的信息预测该选哪个ASR结果
# - 全程按 reference 文本分组做5折OOF，避免相同指令泄漏
# - 训练标签只使用两模型编辑距离不相同的样本
# - 最终优化目标不是分类准确率，而是最终字符编辑距离/CER
# =========================================================

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT / "output"

STEP29_DETAILS = OUTPUT_DIR / "step29_sensevoice_full_details.jsonl"
INFERENCE_DETAILS = OUTPUT_DIR / "final_testA_inference_details_optimized.csv"

MODEL_PATH = OUTPUT_DIR / "dual_asr_selector.joblib"
REPORT_JSON = OUTPUT_DIR / "step30_dual_asr_selector_validation.json"
OOF_CSV = OUTPUT_DIR / "step30_dual_asr_selector_oof.csv"

if not STEP29_DETAILS.exists():
    raise FileNotFoundError(f"缺少文件：{STEP29_DETAILS}")


# =========================================================
# 文本工具
# =========================================================

def normalize_text(text):
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", "", text)
    return text


def normalize_path(path):
    return str(Path(path).resolve()).lower()


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
    """
    轻量字符集合重合度，不用reference。
    """
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
# 读取speaker score（可选）
# =========================================================

speaker_scores = {}

if INFERENCE_DETAILS.exists():
    with INFERENCE_DETAILS.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("command_path")
            score = row.get("speaker_score")
            if raw and score not in ("", None):
                speaker_scores[normalize_path(raw)] = float(score)

print("speaker_score可匹配数量：", len(speaker_scores))


# =========================================================
# 读取Step29详情
# =========================================================

rows = []

with STEP29_DETAILS.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)

        command_path = item["command_path"]
        key = normalize_path(command_path)

        ref = normalize_text(item["reference"])
        para = normalize_text(item["paraformer"])
        sv = normalize_text(item["sensevoice"])

        dp = int(item["para_distance"])
        ds = int(item["sensevoice_distance"])

        if ds < dp:
            winner = 1   # SenseVoice
        elif dp < ds:
            winner = 0   # Paraformer
        else:
            winner = -1  # tie

        rows.append(
            {
                "command_path": command_path,
                "key": key,
                "reference": ref,
                "para": para,
                "sv": sv,
                "dp": dp,
                "ds": ds,
                "winner": winner,
                "speaker_score": speaker_scores.get(key, 0.0),
            }
        )

n = len(rows)

if n == 0:
    raise RuntimeError("Step29详情为空")

print("样本数：", n)

sv_better = sum(r["winner"] == 1 for r in rows)
para_better = sum(r["winner"] == 0 for r in rows)
ties = sum(r["winner"] == -1 for r in rows)

print("SenseVoice更好：", sv_better)
print("Paraformer更好：", para_better)
print("编辑距离相同：", ties)


# =========================================================
# OOF：按reference分组
# =========================================================

groups = np.array([
    r["reference"] if r["reference"] else f"EMPTY_{i}"
    for i, r in enumerate(rows)
])

indices = np.arange(n)

gkf = GroupKFold(n_splits=5)

oof_prob = np.full(n, np.nan, dtype=float)
fold_ids = np.full(n, -1, dtype=int)

for fold, (train_idx, val_idx) in enumerate(
    gkf.split(indices, groups=groups),
    start=1,
):
    # 只用明确有赢家的训练样本
    train_non_tie = np.array([
        i for i in train_idx
        if rows[i]["winner"] in (0, 1)
    ])

    if len(train_non_tie) == 0:
        raise RuntimeError(f"第{fold}折没有非tie训练样本")

    y_train = np.array([
        rows[i]["winner"]
        for i in train_non_tie
    ])

    print(
        f"第{fold}折："
        f"train_non_tie={len(train_non_tie)}，"
        f"val={len(val_idx)}，"
        f"sv_class={int(y_train.sum())}，"
        f"para_class={int((y_train == 0).sum())}"
    )

    para_train = [rows[i]["para"] for i in train_non_tie]
    sv_train = [rows[i]["sv"] for i in train_non_tie]

    para_val = [rows[i]["para"] for i in val_idx]
    sv_val = [rows[i]["sv"] for i in val_idx]

    # 两个ASR分别建字符TF-IDF，避免把两个输出混在一起
    vec_para = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 4),
        min_df=2,
        max_features=12000,
        sublinear_tf=True,
    )

    vec_sv = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 4),
        min_df=2,
        max_features=12000,
        sublinear_tf=True,
    )

    xp_train = vec_para.fit_transform(para_train)
    xs_train = vec_sv.fit_transform(sv_train)

    xp_val = vec_para.transform(para_val)
    xs_val = vec_sv.transform(sv_val)

    numeric_train = csr_matrix(
        np.asarray([
            make_numeric_features(
                rows[i]["para"],
                rows[i]["sv"],
                rows[i]["speaker_score"],
            )
            for i in train_non_tie
        ], dtype=np.float32)
    )

    numeric_val = csr_matrix(
        np.asarray([
            make_numeric_features(
                rows[i]["para"],
                rows[i]["sv"],
                rows[i]["speaker_score"],
            )
            for i in val_idx
        ], dtype=np.float32)
    )

    x_train = hstack(
        [xp_train, xs_train, numeric_train],
        format="csr",
    )

    x_val = hstack(
        [xp_val, xs_val, numeric_val],
        format="csr",
    )

    clf = LogisticRegression(
        C=1.5,
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=20260807 + fold,
    )

    clf.fit(x_train, y_train)

    probs = clf.predict_proba(x_val)[:, 1]

    oof_prob[val_idx] = probs
    fold_ids[val_idx] = fold


if np.isnan(oof_prob).any():
    raise RuntimeError("OOF概率存在缺失")


# =========================================================
# 分类能力（只看非tie）
# =========================================================

non_tie_idx = np.array([
    i for i, r in enumerate(rows)
    if r["winner"] in (0, 1)
])

y_true = np.array([
    rows[i]["winner"]
    for i in non_tie_idx
])

y_prob = oof_prob[non_tie_idx]
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_true, y_pred)

try:
    auc = roc_auc_score(y_true, y_prob)
except Exception:
    auc = None

print()
print("=" * 88)
print("Step30 OOF选择器分类能力（只统计非tie）")
print("=" * 88)
print("非tie样本：", len(non_tie_idx))
print("OOF分类准确率：", acc)
print("OOF ROC-AUC：", auc)


# =========================================================
# CER评估
# =========================================================

total_chars = sum(len(r["reference"]) for r in rows)

para_dist = sum(r["dp"] for r in rows)
sv_dist = sum(r["ds"] for r in rows)
oracle_dist = sum(min(r["dp"], r["ds"]) for r in rows)

para_cer = para_dist / total_chars
sv_cer = sv_dist / total_chars
oracle_cer = oracle_dist / total_chars


def evaluate_threshold(threshold):
    selected_dist = 0
    sv_selected = 0
    para_selected = 0
    correct_winner = 0
    non_tie_total = 0

    for i, r in enumerate(rows):
        choose_sv = oof_prob[i] >= threshold

        if choose_sv:
            selected_dist += r["ds"]
            sv_selected += 1
        else:
            selected_dist += r["dp"]
            para_selected += 1

        if r["winner"] in (0, 1):
            non_tie_total += 1
            if int(choose_sv) == r["winner"]:
                correct_winner += 1

    cer = selected_dist / total_chars

    return {
        "threshold": float(threshold),
        "cer": float(cer),
        "distance": int(selected_dist),
        "sv_selected": int(sv_selected),
        "para_selected": int(para_selected),
        "sv_select_rate": sv_selected / n,
        "winner_accuracy": (
            correct_winner / non_tie_total
            if non_tie_total
            else None
        ),
    }


default_result = evaluate_threshold(0.5)

threshold_results = []

for t in np.arange(0.20, 0.801, 0.01):
    threshold_results.append(
        evaluate_threshold(round(float(t), 2))
    )

best_oof = min(
    threshold_results,
    key=lambda x: (
        x["cer"],
        abs(x["threshold"] - 0.5),
    ),
)


# =========================================================
# 简单规则baseline
# =========================================================

def rule_shorter():
    d = 0
    sv_count = 0
    for r in rows:
        if len(r["sv"]) < len(r["para"]):
            d += r["ds"]
            sv_count += 1
        else:
            d += r["dp"]
    return d / total_chars, sv_count / n


def rule_longer():
    d = 0
    sv_count = 0
    for r in rows:
        if len(r["sv"]) > len(r["para"]):
            d += r["ds"]
            sv_count += 1
        else:
            d += r["dp"]
    return d / total_chars, sv_count / n


shorter_cer, shorter_sv_rate = rule_shorter()
longer_cer, longer_sv_rate = rule_longer()


# =========================================================
# 输出关键结果
# =========================================================

def oracle_capture(selector_cer):
    denominator = sv_cer - oracle_cer
    if denominator <= 0:
        return 0.0
    return (sv_cer - selector_cer) / denominator


print()
print("=" * 88)
print("Step30 OOF双ASR选择结果")
print("=" * 88)

print(f"固定Paraformer CER： {para_cer:.6f} ({para_cer*100:.3f}%)")
print(f"固定SenseVoice CER： {sv_cer:.6f} ({sv_cer*100:.3f}%)")
print(f"Oracle CER：         {oracle_cer:.6f} ({oracle_cer*100:.3f}%)")

print()
print(
    f"OOF选择器 t=0.50： "
    f"CER={default_result['cer']*100:.3f}% "
    f"SenseVoice选择率={default_result['sv_select_rate']*100:.2f}% "
    f"赢家分类准确率={default_result['winner_accuracy']*100:.2f}%"
)

print(
    f"OOF最佳阈值：       "
    f"t={best_oof['threshold']:.2f} "
    f"CER={best_oof['cer']*100:.3f}% "
    f"SenseVoice选择率={best_oof['sv_select_rate']*100:.2f}% "
    f"赢家分类准确率={best_oof['winner_accuracy']*100:.2f}%"
)

print(
    f"相对固定SenseVoice："
    f"{(best_oof['cer'] - sv_cer)*100:+.3f} 个百分点"
)

print(
    f"捕获Oracle剩余空间："
    f"{oracle_capture(best_oof['cer'])*100:.1f}%"
)

print()
print(
    f"简单规则-选更短文本 CER："
    f"{shorter_cer*100:.3f}% "
    f"(SenseVoice选择率{shorter_sv_rate*100:.1f}%)"
)

print(
    f"简单规则-选更长文本 CER："
    f"{longer_cer*100:.3f}% "
    f"(SenseVoice选择率{longer_sv_rate*100:.1f}%)"
)


# =========================================================
# 保存OOF明细
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
        "winner",
        "speaker_score",
        "sensevoice_probability",
        "selected_at_best_threshold",
        "selected_distance_at_best_threshold",
    ]

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for i, r in enumerate(rows):
        choose_sv = oof_prob[i] >= best_oof["threshold"]

        writer.writerow(
            {
                "fold": int(fold_ids[i]),
                "command_path": r["command_path"],
                "reference": r["reference"],
                "paraformer": r["para"],
                "sensevoice": r["sv"],
                "para_distance": r["dp"],
                "sensevoice_distance": r["ds"],
                "winner": (
                    "sensevoice"
                    if r["winner"] == 1
                    else "paraformer"
                    if r["winner"] == 0
                    else "tie"
                ),
                "speaker_score": r["speaker_score"],
                "sensevoice_probability": oof_prob[i],
                "selected_at_best_threshold": (
                    "sensevoice" if choose_sv else "paraformer"
                ),
                "selected_distance_at_best_threshold": (
                    r["ds"] if choose_sv else r["dp"]
                ),
            }
        )


# =========================================================
# 训练最终选择器
# =========================================================

train_idx = non_tie_idx
y_final = np.array([
    rows[i]["winner"]
    for i in train_idx
])

para_train = [rows[i]["para"] for i in train_idx]
sv_train = [rows[i]["sv"] for i in train_idx]

vec_para = TfidfVectorizer(
    analyzer="char",
    ngram_range=(1, 4),
    min_df=2,
    max_features=12000,
    sublinear_tf=True,
)

vec_sv = TfidfVectorizer(
    analyzer="char",
    ngram_range=(1, 4),
    min_df=2,
    max_features=12000,
    sublinear_tf=True,
)

xp = vec_para.fit_transform(para_train)
xs = vec_sv.fit_transform(sv_train)

xn = csr_matrix(
    np.asarray([
        make_numeric_features(
            rows[i]["para"],
            rows[i]["sv"],
            rows[i]["speaker_score"],
        )
        for i in train_idx
    ], dtype=np.float32)
)

x = hstack([xp, xs, xn], format="csr")

clf = LogisticRegression(
    C=1.5,
    class_weight="balanced",
    max_iter=3000,
    solver="liblinear",
    random_state=20260807,
)

clf.fit(x, y_final)

bundle = {
    "vectorizer_para": vec_para,
    "vectorizer_sensevoice": vec_sv,
    "classifier": clf,
    "threshold": best_oof["threshold"],
    "domain_words": DOMAIN_WORDS,
    "feature_version": 1,
    "oof_metrics": {
        "paraformer_cer": para_cer,
        "sensevoice_cer": sv_cer,
        "oracle_cer": oracle_cer,
        "selector_default": default_result,
        "selector_best": best_oof,
        "oof_non_tie_accuracy_0_5": acc,
        "oof_auc": auc,
    },
}

joblib.dump(bundle, MODEL_PATH)


# =========================================================
# 保存报告
# =========================================================

report = {
    "sample_count": n,
    "non_tie_count": int(len(non_tie_idx)),
    "sensevoice_better_count": sv_better,
    "paraformer_better_count": para_better,
    "tie_count": ties,

    "paraformer_cer": para_cer,
    "sensevoice_cer": sv_cer,
    "oracle_cer": oracle_cer,

    "oof_accuracy_non_tie_0_5": acc,
    "oof_auc_non_tie": auc,

    "selector_default_0_5": default_result,
    "selector_best_oof_threshold": best_oof,

    "selector_vs_sensevoice_pp": (
        best_oof["cer"] - sv_cer
    ) * 100,

    "oracle_capture_ratio": oracle_capture(
        best_oof["cer"]
    ),

    "shorter_text_rule": {
        "cer": shorter_cer,
        "sv_select_rate": shorter_sv_rate,
    },

    "longer_text_rule": {
        "cer": longer_cer,
        "sv_select_rate": longer_sv_rate,
    },

    "threshold_grid": threshold_results,
}

with REPORT_JSON.open("w", encoding="utf-8") as f:
    json.dump(
        report,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("=" * 88)
print("保存完成")
print("=" * 88)
print("最终选择器：", MODEL_PATH)
print("OOF报告：", REPORT_JSON)
print("OOF明细：", OOF_CSV)

