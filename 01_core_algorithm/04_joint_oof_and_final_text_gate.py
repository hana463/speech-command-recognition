
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
# 选择性双ASR + 文本门控 联合OOF优化
#
# 目的：
# 1. 比较多个 SenseVoice second-pass router threshold
# 2. 每个 threshold 都重新生成“严格OOF选择性ASR文本”
# 3. 每个 threshold 都重新做文本门控5折OOF
# 4. 联合搜索 low / high / text-p
# 5. 同时统计：
#       CER / RR / recognition score
#       Paraformer调用率
#       SenseVoice调用率
#       总ASR模型调用次数/样本

# =========================================================

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data" / "testA"
OUTPUT_DIR = PROJECT / "output"

POS_JSONL = DATA_DIR / "pos.jsonl"
NEG_JSONL = DATA_DIR / "neg.jsonl"

PARA_CACHE = OUTPUT_DIR / "asr_hotword_best_full_cache.jsonl"
SV_CACHE = OUTPUT_DIR / "asr_sensevoice_full_cache.jsonl"

STEP30_OOF = OUTPUT_DIR / "step30_dual_asr_selector_oof.csv"
STEP33_OOF = OUTPUT_DIR / "step33_second_pass_router_oof.csv"

SELECTOR_MODEL = OUTPUT_DIR / "dual_asr_selector.joblib"
ROUTER_MODEL = OUTPUT_DIR / "dual_asr_second_pass_router.joblib"

SPEAKER_DETAIL_CSV = OUTPUT_DIR / "final_testA_inference_details_optimized.csv"

OUT_JSON = OUTPUT_DIR / "step34_joint_selective_pipeline.json"
OUT_CSV = OUTPUT_DIR / "step34_joint_selective_pipeline_candidates.csv"

FINAL_TEXT_GATE_MODEL = OUTPUT_DIR / "selective_dual_text_gate_model.joblib"
FINAL_POLICY = OUTPUT_DIR / "selective_dual_pipeline_policy.json"
FINAL_FULL_CACHE = OUTPUT_DIR / "asr_selective_dual_final_full_cache.jsonl"


# =========================================================
# 需要联合比较的 Router threshold
# =========================================================

ROUTER_THRESHOLDS = [
    0.23,   # ~69% SV calls
    0.28,   # ~59%
    0.35,   # ~47%
    0.37,   # ~44%
    0.41,   # ~39%
    0.43,   # ~36%
    0.47,   # ~30%
]

SELECTOR_THRESHOLD = 0.32


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
            if line:
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

            cache[key] = normalize_text(
                item.get("text", "")
            )

    return cache


# =========================================================
# Selector特征（必须与Step30一致）
# =========================================================

def chinese_ratio(text):
    if not text:
        return 0.0

    n = sum(
        "\u4e00" <= ch <= "\u9fff"
        for ch in text
    )

    return n / len(text)


def ascii_ratio(text):
    if not text:
        return 0.0

    n = sum(
        ord(ch) < 128
        for ch in text
    )

    return n / len(text)


def digit_ratio(text):
    if not text:
        return 0.0

    n = sum(
        ch.isdigit()
        for ch in text
    )

    return n / len(text)


def repeated_ratio(text):
    if len(text) < 2:
        return 0.0

    repeats = sum(
        text[i] == text[i - 1]
        for i in range(1, len(text))
    )

    return repeats / (len(text) - 1)


def overlap_ratio(a, b):
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    sa = set(a)
    sb = set(b)

    union = len(sa | sb)

    return (
        len(sa & sb) / union
        if union
        else 0.0
    )


def prefix_ratio(a, b):
    if not a or not b:
        return 0.0

    n = 0

    for x, y in zip(a, b):
        if x != y:
            break
        n += 1

    return n / max(len(a), len(b), 1)


def suffix_ratio(a, b):
    if not a or not b:
        return 0.0

    n = 0

    for x, y in zip(
        reversed(a),
        reversed(b),
    ):
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
    return sum(
        word in text
        for word in DOMAIN_WORDS
    )


def selector_numeric_features(
    para,
    sv,
    speaker_score,
):
    lp = len(para)
    ls = len(sv)

    inter_dist = editdistance.eval(
        para,
        sv,
    )

    inter_norm = (
        inter_dist
        / max(lp, ls, 1)
    )

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
        overlap_ratio(para, sv),
        prefix_ratio(para, sv),
        suffix_ratio(para, sv),
        domain_hits(para),
        domain_hits(sv),
        domain_hits(para) - domain_hits(sv),
        speaker_score,
    ]


# =========================================================
# Router特征（必须与Step33一致）
# =========================================================

def router_numeric_features(
    text,
    speaker_score,
):
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

for required in [
    POS_JSONL,
    NEG_JSONL,
    PARA_CACHE,
    SV_CACHE,
    STEP30_OOF,
    STEP33_OOF,
    SELECTOR_MODEL,
    ROUTER_MODEL,
    SPEAKER_DETAIL_CSV,
]:
    if not required.exists():
        raise FileNotFoundError(
            f"缺少文件：{required}"
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

print("总样本：", len(all_keys))
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

missing = [
    k for k in all_keys
    if k not in speaker_scores
]

if missing:
    raise RuntimeError(
        f"speaker_score缺少{len(missing)}条"
    )

print(
    "speaker_score：",
    len(speaker_scores),
)


# =========================================================
# ASR缓存
# =========================================================

para_cache = load_cache(
    PARA_CACHE
)

sv_cache = load_cache(
    SV_CACHE
)

for name, cache in [
    ("Paraformer", para_cache),
    ("SenseVoice", sv_cache),
]:
    missing = [
        k for k in all_keys
        if k not in cache
    ]

    if missing:
        raise RuntimeError(
            f"{name}缓存缺少{len(missing)}条"
        )

    print(
        f"{name}缓存：",
        len(cache),
    )


# =========================================================
# OOF selector / router probabilities（正样本）
# =========================================================

selector_oof = {}

with STEP30_OOF.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:
    reader = csv.DictReader(f)

    for row in reader:
        selector_oof[
            normalize_path(
                row["command_path"]
            )
        ] = float(
            row[
                "sensevoice_probability"
            ]
        )


router_oof = {}

with STEP33_OOF.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:
    reader = csv.DictReader(f)

    for row in reader:
        router_oof[
            normalize_path(
                row["command_path"]
            )
        ] = float(
            row[
                "router_probability"
            ]
        )

print(
    "正样本OOF selector：",
    len(selector_oof),
)
print(
    "正样本OOF router：",
    len(router_oof),
)


# =========================================================
# 最终 selector 概率（1838全量）
# =========================================================

selector_bundle = joblib.load(
    SELECTOR_MODEL
)

selector_vec_para = (
    selector_bundle[
        "vectorizer_para"
    ]
)

selector_vec_sv = (
    selector_bundle[
        "vectorizer_sensevoice"
    ]
)

selector_clf = (
    selector_bundle[
        "classifier"
    ]
)

para_texts = [
    para_cache[k]
    for k in all_keys
]

sv_texts = [
    sv_cache[k]
    for k in all_keys
]

xp = selector_vec_para.transform(
    para_texts
)

xs = selector_vec_sv.transform(
    sv_texts
)

xn = csr_matrix(
    np.asarray(
        [
            selector_numeric_features(
                para_texts[i],
                sv_texts[i],
                speaker_scores[
                    all_keys[i]
                ],
            )
            for i in range(
                len(all_keys)
            )
        ],
        dtype=np.float32,
    )
)

x_selector = hstack(
    [xp, xs, xn],
    format="csr",
)

selector_full_probs = (
    selector_clf.predict_proba(
        x_selector
    )[:, 1]
)

selector_full_map = {
    key: float(prob)
    for key, prob
    in zip(
        all_keys,
        selector_full_probs,
    )
}


# =========================================================
# 最终 Router 概率（1838全量）
# =========================================================

router_bundle = joblib.load(
    ROUTER_MODEL
)

router_vec = (
    router_bundle["vectorizer"]
)

router_clf = (
    router_bundle["classifier"]
)

xr_text = router_vec.transform(
    para_texts
)

xr_num = csr_matrix(
    np.asarray(
        [
            router_numeric_features(
                para_texts[i],
                speaker_scores[
                    all_keys[i]
                ],
            )
            for i in range(
                len(all_keys)
            )
        ],
        dtype=np.float32,
    )
)

x_router = hstack(
    [
        xr_text,
        xr_num,
    ],
    format="csr",
)

router_full_probs = (
    router_clf.predict_proba(
        x_router
    )[:, 1]
)

router_full_map = {
    key: float(prob)
    for key, prob
    in zip(
        all_keys,
        router_full_probs,
    )
}


# =========================================================
# 构造候选ASR文本
# =========================================================

def build_candidate_texts(
    router_threshold,
    use_oof_for_positive=True,
):
    rows = []

    for key in all_keys:
        para = para_cache[key]
        sv = sv_cache[key]

        if (
            use_oof_for_positive
            and labels[key] == 1
        ):
            if key not in router_oof:
                raise RuntimeError(
                    f"正样本缺router OOF：{key}"
                )

            if key not in selector_oof:
                raise RuntimeError(
                    f"正样本缺selector OOF：{key}"
                )

            router_prob = router_oof[key]
            selector_prob = selector_oof[key]
            prob_source = "oof"

        else:
            router_prob = (
                router_full_map[key]
            )

            selector_prob = (
                selector_full_map[key]
            )

            prob_source = "full"

        second_call = (
            router_prob
            >= router_threshold
        )

        if second_call:
            if (
                selector_prob
                >= SELECTOR_THRESHOLD
            ):
                text = sv
                selected_model = (
                    "sensevoice"
                )
            else:
                text = para
                selected_model = (
                    "paraformer_after_sv"
                )

        else:
            text = para
            selected_model = (
                "paraformer_only"
            )

        ref = references[key]

        distance = (
            editdistance.eval(
                ref,
                text,
            )
            if labels[key] == 1
            else 0
        )

        rows.append(
            {
                "key": key,
                "split": splits[key],
                "label": labels[key],
                "reference": ref,
                "asr_text": text,
                "asr_distance": distance,
                "asr_empty": text == "",
                "speaker_score": (
                    speaker_scores[key]
                ),
                "router_probability": (
                    router_prob
                ),
                "selector_probability": (
                    selector_prob
                ),
                "second_asr_call": (
                    second_call
                ),
                "selected_model": (
                    selected_model
                ),
                "probability_source": (
                    prob_source
                ),
            }
        )

    return rows


# =========================================================
# 文本门控 OOF
# =========================================================

def make_text_gate_oof(rows):

    n = len(rows)

    y = np.asarray(
        [
            r["label"]
            for r in rows
        ],
        dtype=np.int32,
    )

    groups = np.asarray(
        [
            model_text(
                r["asr_text"]
            )
            for r in rows
        ],
        dtype=object,
    )

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

    for fold, (
        train_idx,
        val_idx,
    ) in enumerate(
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

        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 4),
            min_df=2,
            max_features=16000,
            sublinear_tf=True,
        )

        x_train = (
            vectorizer.fit_transform(
                [
                    model_text(t)
                    for t in train_texts
                ]
            )
        )

        classifier = LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=(
                20260807 + fold
            ),
        )

        classifier.fit(
            x_train,
            y[train_idx],
        )

        x_val = vectorizer.transform(
            [
                model_text(t)
                for t in val_texts
            ]
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

    if np.isnan(oof_prob).any():
        raise RuntimeError(
            "text gate OOF概率不完整"
        )

    pred = (
        oof_prob >= 0.5
    ).astype(np.int32)

    acc = accuracy_score(
        y,
        pred,
    )

    try:
        auc = roc_auc_score(
            y,
            oof_prob,
        )
    except Exception:
        auc = None

    return oof_prob, acc, auc


# =========================================================
# 门控策略搜索
# =========================================================

LOW_VALUES = [
    round(x / 1000, 3)
    for x in range(-50, 101, 5)
]

HIGH_VALUES = [
    round(x / 1000, 3)
    for x in range(300, 461, 5)
]

P_VALUES = [
    round(x / 100, 2)
    for x in range(40, 91, 2)
]


def search_gate(
    rows,
    text_prob,
):
    y = np.asarray(
        [
            r["label"]
            for r in rows
        ],
        dtype=np.int32,
    )

    speaker = np.asarray(
        [
            r["speaker_score"]
            for r in rows
        ],
        dtype=np.float64,
    )

    ref_len = np.asarray(
        [
            len(r["reference"])
            if r["label"] == 1
            else 0
            for r in rows
        ],
        dtype=np.int32,
    )

    asr_dist = np.asarray(
        [
            r["asr_distance"]
            for r in rows
        ],
        dtype=np.int32,
    )

    asr_empty = np.asarray(
        [
            r["asr_empty"]
            for r in rows
        ],
        dtype=bool,
    )

    second_call = np.asarray(
        [
            r["second_asr_call"]
            for r in rows
        ],
        dtype=bool,
    )

    pos = y == 1
    neg = y == 0

    total_chars = int(
        np.sum(
            ref_len[pos]
        )
    )

    results = []

    for low in LOW_VALUES:
        low_region = (
            speaker < low
        )

        para_call = (
            ~low_region
        )

        # Router在Paraformer之后，
        # 只有真正进入ASR区才会发生第二模型调用
        effective_second = (
            para_call
            & second_call
        )

        para_call_rate = float(
            np.mean(para_call)
        )

        second_call_rate = float(
            np.mean(
                effective_second
            )
        )

        model_calls_per_sample = (
            para_call_rate
            + second_call_rate
        )

        for high in HIGH_VALUES:

            if low >= high:
                continue

            high_region = (
                speaker >= high
            )

            middle = (
                (~low_region)
                & (~high_region)
            )

            for p in P_VALUES:

                accepted = (
                    high_region
                    | (
                        middle
                        & (
                            text_prob >= p
                        )
                    )
                )

                final_dist = np.where(
                    accepted[pos],
                    asr_dist[pos],
                    ref_len[pos],
                )

                total_dist = int(
                    np.sum(
                        final_dist
                    )
                )

                cer = (
                    total_dist
                    / total_chars
                )

                neg_empty = (
                    (~accepted[neg])
                    | asr_empty[neg]
                )

                rr = float(
                    np.mean(
                        neg_empty
                    )
                )

                recognition = (
                    0.5 * (1.0 - cer)
                    + 0.5 * rr
                )

                text_gate_rate = float(
                    np.mean(middle)
                )

                results.append(
                    {
                        "low": low,
                        "high": high,
                        "p": p,
                        "cer": float(cer),
                        "rr": rr,
                        "recognition_score": (
                            float(
                                recognition
                            )
                        ),
                        "paraformer_call_rate": (
                            para_call_rate
                        ),
                        "sensevoice_call_rate": (
                            second_call_rate
                        ),
                        "asr_model_calls_per_sample": (
                            model_calls_per_sample
                        ),
                        "text_gate_call_rate": (
                            text_gate_rate
                        ),
                        "total_edit_distance": (
                            total_dist
                        ),
                    }
                )

    best_recognition = max(
        results,
        key=lambda r: (
            r[
                "recognition_score"
            ],
            -r["cer"],
            r["rr"],
        ),
    )

    # 识别分距最优<=0.0015时，
    # 选择最少模型调用
    robust_pool = [
        r for r in results
        if (
            best_recognition[
                "recognition_score"
            ]
            - r[
                "recognition_score"
            ]
            <= 0.0015
        )
    ]

    robust = min(
        robust_pool,
        key=lambda r: (
            r[
                "asr_model_calls_per_sample"
            ],
            r["cer"],
            -r["rr"],
        ),
    )

    return (
        best_recognition,
        robust,
        results,
    )


# =========================================================
# 逐个 Router threshold 联合验证
# =========================================================

candidate_summaries = []
candidate_grids = {}

for router_threshold in ROUTER_THRESHOLDS:

    print()
    print("=" * 96)
    print(
        "联合验证 Router threshold =",
        router_threshold,
    )
    print("=" * 96)

    rows = build_candidate_texts(
        router_threshold,
        use_oof_for_positive=True,
    )

    pos_rows = [
        r for r in rows
        if r["label"] == 1
    ]

    total_chars = sum(
        len(r["reference"])
        for r in pos_rows
    )

    pure_dist = sum(
        r["asr_distance"]
        for r in pos_rows
    )

    pure_cer = (
        pure_dist / total_chars
    )

    raw_second_rate_pos = (
        sum(
            r["second_asr_call"]
            for r in pos_rows
        )
        / len(pos_rows)
    )

    print(
        f"选择性ASR严格OOF CER："
        f"{pure_cer*100:.3f}%"
    )
    print(
        f"正样本SenseVoice原始调用率："
        f"{raw_second_rate_pos*100:.2f}%"
    )

    text_prob, gate_acc, gate_auc = (
        make_text_gate_oof(rows)
    )

    print(
        "文本门控OOF Accuracy：",
        gate_acc,
    )
    print(
        "文本门控OOF AUC：",
        gate_auc,
    )

    (
        best_recognition,
        robust,
        grid,
    ) = search_gate(
        rows,
        text_prob,
    )

    candidate_grids[
        str(router_threshold)
    ] = grid

    summary = {
        "router_threshold": (
            router_threshold
        ),
        "pure_oof_asr_cer": (
            pure_cer
        ),
        "raw_positive_sv_call_rate": (
            raw_second_rate_pos
        ),
        "text_gate_oof_accuracy": (
            float(gate_acc)
        ),
        "text_gate_oof_auc": (
            None
            if gate_auc is None
            else float(gate_auc)
        ),
        "recognition_best": (
            best_recognition
        ),
        "robust": robust,
    }

    candidate_summaries.append(
        summary
    )

    print(
        "识别最高：",
        best_recognition,
    )
    print(
        "同分附近最省调用：",
        robust,
    )


# =========================================================
# Router threshold 跨候选选择
# =========================================================

global_best = max(
    candidate_summaries,
    key=lambda x: (
        x[
            "recognition_best"
        ][
            "recognition_score"
        ],
        -x[
            "recognition_best"
        ][
            "asr_model_calls_per_sample"
        ],
    ),
)

best_recognition_value = (
    global_best[
        "recognition_best"
    ][
        "recognition_score"
    ]
)

# 允许距全局最佳 recognition <= 0.002，
# 选最少总ASR模型调用
competition_pool = []

for item in candidate_summaries:
    for mode in [
        "recognition_best",
        "robust",
    ]:
        policy = item[mode]

        if (
            best_recognition_value
            - policy[
                "recognition_score"
            ]
            <= 0.002
        ):
            competition_pool.append(
                {
                    "router_threshold": (
                        item[
                            "router_threshold"
                        ]
                    ),
                    "pure_oof_asr_cer": (
                        item[
                            "pure_oof_asr_cer"
                        ]
                    ),
                    "policy_mode": mode,
                    **policy,
                }
            )

competition_recommended = min(
    competition_pool,
    key=lambda r: (
        r[
            "asr_model_calls_per_sample"
        ],
        r["cer"],
        -r["rr"],
    ),
)


# =========================================================
# 输出总表
# =========================================================

print()
print("=" * 106)
print("Step34 Router threshold 联合结果")
print("=" * 106)

for item in candidate_summaries:

    r = item[
        "recognition_best"
    ]

    print(
        f"router={item['router_threshold']:.2f} | "
        f"PureCER={item['pure_oof_asr_cer']*100:6.3f}% | "
        f"FinalCER={r['cer']*100:6.3f}% "
        f"RR={r['rr']*100:6.3f}% "
        f"Rec={r['recognition_score']:.6f} | "
        f"Para={r['paraformer_call_rate']*100:5.2f}% "
        f"SV={r['sensevoice_call_rate']*100:5.2f}% "
        f"Calls/sample={r['asr_model_calls_per_sample']:.3f}"
    )

print()
print("=" * 106)
print("Step34 最终推荐")
print("=" * 106)

print("联合OOF识别分最高：")
print(global_best)

print()
print("比赛效率/识别稳健推荐：")
print(competition_recommended)


# =========================================================
# 用最终模型输出训练真正隐藏B文本门控
# =========================================================

final_router_threshold = float(
    competition_recommended[
        "router_threshold"
    ]
)

final_low = float(
    competition_recommended[
        "low"
    ]
)

final_high = float(
    competition_recommended[
        "high"
    ]
)

final_p = float(
    competition_recommended[
        "p"
    ]
)

full_rows = build_candidate_texts(
    final_router_threshold,
    use_oof_for_positive=False,
)

full_texts = [
    r["asr_text"]
    for r in full_rows
]

full_y = np.asarray(
    [
        r["label"]
        for r in full_rows
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

joblib.dump(
    {
        "vectorizer": final_vectorizer,
        "classifier": final_classifier,
        "training_router_threshold": (
            final_router_threshold
        ),
        "selector_threshold": (
            SELECTOR_THRESHOLD
        ),
        "feature_version": 1,
    },
    FINAL_TEXT_GATE_MODEL,
)


# =========================================================
# 保存最终full cache
# =========================================================

with FINAL_FULL_CACHE.open(
    "w",
    encoding="utf-8",
) as f:
    for row in full_rows:
        f.write(
            json.dumps(
                {
                    "command_key": (
                        row["key"]
                    ),
                    "command_path": (
                        command_paths[
                            row["key"]
                        ]
                    ),
                    "split": (
                        row["split"]
                    ),
                    "text": (
                        row["asr_text"]
                    ),
                    "second_asr_call": (
                        row[
                            "second_asr_call"
                        ]
                    ),
                    "selected_model": (
                        row[
                            "selected_model"
                        ]
                    ),
                    "router_probability": (
                        row[
                            "router_probability"
                        ]
                    ),
                    "selector_probability": (
                        row[
                            "selector_probability"
                        ]
                    ),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


# =========================================================
# 最终policy
# =========================================================

policy_payload = {
    "mode": (
        "selective_dual_asr"
    ),

    "router_threshold": (
        final_router_threshold
    ),

    "selector_threshold": (
        SELECTOR_THRESHOLD
    ),

    "low_threshold": (
        final_low
    ),

    "high_threshold": (
        final_high
    ),

    "text_probability_threshold": (
        final_p
    ),

    "oof_metrics": {
        "pure_oof_asr_cer": (
            competition_recommended[
                "pure_oof_asr_cer"
            ]
        ),
        "final_oof_cer": (
            competition_recommended[
                "cer"
            ]
        ),
        "final_oof_rr": (
            competition_recommended[
                "rr"
            ]
        ),
        "final_oof_recognition_score": (
            competition_recommended[
                "recognition_score"
            ]
        ),
        "paraformer_call_rate": (
            competition_recommended[
                "paraformer_call_rate"
            ]
        ),
        "sensevoice_call_rate": (
            competition_recommended[
                "sensevoice_call_rate"
            ]
        ),
        "asr_model_calls_per_sample": (
            competition_recommended[
                "asr_model_calls_per_sample"
            ]
        ),
    },

    "alternatives": (
        candidate_summaries
    ),

    "notes": [
        (
            "正样本Router与Selector策略验证均使用OOF概率"
        ),
        (
            "文本门控同样使用5折StratifiedGroupKFold OOF"
        ),
        (
            "负样本未参与Router/Selector标签训练，使用最终模型概率"
        ),
        (
            "最终隐藏B文本门控模型使用最终Router/Selector输出全量训练"
        ),
    ],
}

with FINAL_POLICY.open(
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
# 保存候选CSV / JSON
# =========================================================

flat_rows = []

for item in candidate_summaries:

    for mode in [
        "recognition_best",
        "robust",
    ]:
        p = item[mode]

        flat_rows.append(
            {
                "router_threshold": (
                    item[
                        "router_threshold"
                    ]
                ),
                "pure_oof_asr_cer": (
                    item[
                        "pure_oof_asr_cer"
                    ]
                ),
                "raw_positive_sv_call_rate": (
                    item[
                        "raw_positive_sv_call_rate"
                    ]
                ),
                "text_gate_oof_accuracy": (
                    item[
                        "text_gate_oof_accuracy"
                    ]
                ),
                "text_gate_oof_auc": (
                    item[
                        "text_gate_oof_auc"
                    ]
                ),
                "policy_mode": mode,
                **p,
            }
        )

with OUT_CSV.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            flat_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(flat_rows)


with OUT_JSON.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "candidate_summaries": (
                candidate_summaries
            ),
            "global_best": (
                global_best
            ),
            "competition_recommended": (
                competition_recommended
            ),
            "final_policy_file": str(
                FINAL_POLICY
            ),
            "final_text_gate_model": str(
                FINAL_TEXT_GATE_MODEL
            ),
            "final_full_cache": str(
                FINAL_FULL_CACHE
            ),
        },
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("=" * 106)
print("Step34 保存完成")
print("=" * 106)
print(
    "联合报告：",
    OUT_JSON,
)
print(
    "候选表：",
    OUT_CSV,
)
print(
    "最终文本门控：",
    FINAL_TEXT_GATE_MODEL,
)
print(
    "最终策略：",
    FINAL_POLICY,
)
print(
    "最终选择性ASR缓存：",
    FINAL_FULL_CACHE,
)


