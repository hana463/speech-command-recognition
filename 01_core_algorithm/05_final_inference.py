

import argparse
import csv
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import editdistance
import joblib
import numpy as np
import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from modelscope.pipelines import pipeline
from scipy.sparse import csr_matrix, hstack
from tqdm import tqdm


# =========================================================
# 最终结构：
#
# Wake + Command
#      ↓
# CAM++ speaker score
#      ↓
# score < low ------------------------------> 拒绝
#      ↓
# Paraformer + 51 hotwords
#      ↓
# Second-pass Router
#      ├─ prob < router_threshold ----------> 保留Paraformer
#      └─ prob >= router_threshold
#                    ↓
#              SenseVoiceSmall
#                    ↓
#              Dual-ASR Selector
#                    ↓
#              选 Paraformer / SenseVoice
#      ↓
# score >= high ----------------------------> 直接接受ASR文本
#      ↓
# Text Gate
#      ├─ prob >= p ------------------------> 接受ASR文本
#      └─ prob < p -------------------------> 拒绝
#
# 默认读取联合OOF优化阶段生成的最终策略：
# router=0.43 / selector=0.32 / low=0.0 / high=0.37 / p=0.78
#
# 注意：
# - TestA用于本地验证；TestB时只需修改 input_jsonl_paths。
# =========================================================


# =========================================================
# 1. 路径 / 命令行
# =========================================================

project_path = Path(__file__).resolve().parents[1]
default_output_dir = project_path / "output"

parser = argparse.ArgumentParser(
    description="XH-202615 最终选择性双ASR推理"
)

parser.add_argument(
    "--data-dir",
    type=Path,
    default=project_path / "data" / "testA",
    help="测试数据目录，例如 data/testA 或 data/testB",
)

parser.add_argument(
    "--input-jsonl",
    nargs="*",
    default=None,
    help=(
        "输入JSONL，可传多个。"
        "若不传：优先自动使用 data-dir 下的 pos.jsonl + neg.jsonl；"
        "否则使用该目录全部 *.jsonl"
    ),
)

parser.add_argument(
    "--output-json",
    type=Path,
    default=default_output_dir / "final_result_step37.json",
    help="最终结果JSON路径",
)

parser.add_argument(
    "--detail-csv",
    type=Path,
    default=default_output_dir / "final_inference_details_step37.csv",
    help="详细推理日志CSV路径",
)

parser.add_argument(
    "--submission",
    action="store_true",
    help=(
        "提交模式：结果JSON不加入local_rr/"
        "local_recognition_score等本地诊断字段。"
        "若数据本身带label，逐条label/cer及final_cer仍按赛题示例保留。"
    ),
)

args = parser.parse_args()

data_path = args.data_dir.resolve()
output_path = args.output_json.parent.resolve()
output_path.mkdir(parents=True, exist_ok=True)

if args.input_jsonl:
    input_jsonl_paths = []

    for raw in args.input_jsonl:
        p = Path(raw)

        if not p.is_absolute():
            candidate_in_data = data_path / p

            if candidate_in_data.exists():
                p = candidate_in_data
            else:
                p = project_path / p

        input_jsonl_paths.append(
            p.resolve()
        )
else:
    pos_jsonl = data_path / "pos.jsonl"
    neg_jsonl = data_path / "neg.jsonl"

    if pos_jsonl.exists() and neg_jsonl.exists():
        input_jsonl_paths = [
            pos_jsonl,
            neg_jsonl,
        ]
    else:
        input_jsonl_paths = sorted(
            data_path.glob("*.jsonl")
        )

        if not input_jsonl_paths:
            raise FileNotFoundError(
                f"在 {data_path} 下没有找到JSONL。"
                "请使用 --input-jsonl 显式指定。"
            )

policy_path = (
    default_output_dir
    / "selective_dual_pipeline_policy.json"
)

text_gate_model_path = (
    default_output_dir
    / "selective_dual_text_gate_model.joblib"
)

router_model_path = (
    default_output_dir
    / "dual_asr_second_pass_router.joblib"
)

selector_model_path = (
    default_output_dir
    / "dual_asr_selector.joblib"
)

hotword_txt_path = (
    default_output_dir
    / "hotwords_clean.txt"
)

result_json_path = args.output_json.resolve()
detail_csv_path = args.detail_csv.resolve()

result_json_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

detail_csv_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

# id继续优先保留JSON里“识别音频”字段的原始相对路径/文件名。
# 赛题文档要求id为测试音频名字。
ID_USE_EXTENSION = True

# TestA本地验证可以输出额外诊断字段；
# --submission 时去掉非官方的 local_rr / local_recognition_score。
INCLUDE_LOCAL_METRICS_WHEN_AVAILABLE = True
SUBMISSION_MODE = bool(args.submission)

print("=" * 72)
print("Step37 输入配置")
print("=" * 72)
print("data-dir：", data_path)
print("input-jsonl：")
for p in input_jsonl_paths:
    print("  ", p)
print("output-json：", result_json_path)
print("detail-csv：", detail_csv_path)
print("submission：", SUBMISSION_MODE)


# =========================================================
# 2. 热词
# =========================================================

DEFAULT_HOTWORDS = [
    "空调",
    "制冷",
    "制热",
    "除湿",
    "抽湿",
    "新风",
    "送风",
    "自动模式",
    "睡眠模式",
    "节能模式",
    "防直吹",
    "上下扫风",
    "左右扫风",
    "扫风",
    "风速",
    "风量",
    "风向",
    "出风口",
    "温度",
    "控温",
    "显示屏",
    "灯光",
    "亮度",
    "色温",
    "冷色调",
    "暖色调",
    "窗帘",
    "客厅",
    "厨房",
    "油烟机",
    "烟机",
    "滤网",
    "调高",
    "调低",
    "调大",
    "调小",
    "十六度",
    "十七度",
    "十八度",
    "十九度",
    "二十度",
    "二十一度",
    "二十二度",
    "二十三度",
    "二十四度",
    "二十五度",
    "二十六度",
    "二十七度",
    "二十八度",
    "二十九度",
    "三十度",
]

if hotword_txt_path.exists():
    clean_hotwords = [
        line.strip()
        for line in hotword_txt_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    print(
        "从hotwords_clean.txt读取热词：",
        len(clean_hotwords),
    )
else:
    clean_hotwords = DEFAULT_HOTWORDS
    print(
        "使用代码内默认热词：",
        len(clean_hotwords),
    )

hotword_string = " ".join(
    clean_hotwords
)


# =========================================================
# 3. 文本工具
# =========================================================

def remove_symbol_other(text):
    """
    SenseVoice可能保留🎼/emoji等事件符号。
    比赛识别文本不应包含这些符号。
    """
    return "".join(
        ch
        for ch in text
        if unicodedata.category(ch) != "So"
    )


def normalize_text(
    text,
    sensevoice=False,
):
    if text is None:
        return ""

    text = str(text).strip()

    if sensevoice:
        text = remove_symbol_other(text)

    text = re.sub(
        r"\s+",
        "",
        text,
    )

    text = re.sub(
        r"[，。！？、；：“”‘’（）()【】\[\],.!?;:'\"—\-]",
        "",
        text,
    )

    return text


def get_model_text(text):
    text = normalize_text(text)
    return (
        text
        if text
        else "空文本占位"
    )


# =========================================================
# 4. JSON字段 / 路径
# =========================================================

def get_first_value(
    item,
    possible_keys,
    field_description,
    required=True,
):
    for key in possible_keys:
        if (
            key in item
            and item[key] is not None
        ):
            return item[key]

    if required:
        raise KeyError(
            f"样本中找不到{field_description}字段。"
            f"尝试过：{possible_keys}；"
            f"当前字段：{list(item.keys())}"
        )

    return None


wake_audio_keys = [
    "唤醒音频",
    "唤醒音频名字",
    "唤醒语音",
    "注册音频",
    "wake_audio",
    "enroll_audio",
    "reference_audio",
]

command_audio_keys = [
    "识别音频",
    "识别音频名字",
    "指令音频",
    "待识别音频",
    "测试音频",
    "command_audio",
    "test_audio",
    "audio",
]

label_keys = [
    "识别文本",
    "识别文本名字",
    "识别标签",
    "待识别文本",
    "label",
    "text",
]


def resolve_audio_path(
    raw_path,
    jsonl_path,
):
    raw_path = Path(
        str(raw_path)
    )

    if (
        raw_path.is_absolute()
        and raw_path.exists()
    ):
        return raw_path.resolve()

    candidates = [
        jsonl_path.parent / raw_path,
        data_path / raw_path,
        project_path / raw_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"找不到音频：{raw_path}\n尝试：\n"
        + "\n".join(
            str(p)
            for p in candidates
        )
    )


# =========================================================
# 5. 读取样本
# =========================================================

samples = []

for jsonl_path in input_jsonl_paths:

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"找不到输入文件：{jsonl_path}"
        )

    with jsonl_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            wake_raw = get_first_value(
                item,
                wake_audio_keys,
                "唤醒音频",
            )

            command_raw = get_first_value(
                item,
                command_audio_keys,
                "识别音频",
            )

            label_raw = get_first_value(
                item,
                label_keys,
                "识别文本",
                required=False,
            )

            wake_path = resolve_audio_path(
                wake_raw,
                jsonl_path,
            )

            command_path = resolve_audio_path(
                command_raw,
                jsonl_path,
            )

            # 与fixed-id版Step18一致：
            # 优先保留识别音频字段里的相对路径，避免pos/neg同名碰撞。
            raw_id_text = (
                str(command_raw)
                .strip()
                .replace("\\", "/")
            )

            raw_id_path = Path(
                str(command_raw)
            )

            if raw_id_path.is_absolute():
                try:
                    relative_id_path = (
                        command_path.relative_to(
                            data_path.resolve()
                        )
                    )
                    raw_id_text = (
                        relative_id_path
                        .as_posix()
                    )
                except ValueError:
                    raw_id_text = (
                        command_path.name
                    )
            else:
                while raw_id_text.startswith(
                    "./"
                ):
                    raw_id_text = (
                        raw_id_text[2:]
                    )

            if ID_USE_EXTENSION:
                sample_id = raw_id_text
            else:
                sample_id = str(
                    Path(raw_id_text).with_suffix("")
                ).replace("\\", "/")

            label_available = any(
                key in item
                for key in label_keys
            )

            reference = (
                normalize_text(label_raw)
                if label_raw is not None
                else ""
            )

            samples.append(
                {
                    "source_file": (
                        jsonl_path.name
                    ),
                    "source_line": (
                        line_number
                    ),
                    "sample_id": (
                        sample_id
                    ),
                    "wake_path": (
                        wake_path
                    ),
                    "command_path": (
                        command_path
                    ),
                    "reference": (
                        reference
                    ),
                    "label_available": (
                        label_available
                    ),
                }
            )


print(
    "待推理样本总数：",
    len(samples),
)

id_counter = Counter(
    s["sample_id"]
    for s in samples
)

duplicates = [
    sample_id
    for sample_id, count
    in id_counter.items()
    if count > 1
]

if duplicates:
    print(
        "\n检测到重复提交ID，前10个："
    )

    for duplicate_id in duplicates[:10]:
        print(
            "  ",
            duplicate_id,
        )

    raise RuntimeError(
        "最终结果ID仍存在重复，"
        "请检查识别音频字段。"
    )


# =========================================================
# 6. 加载Step34策略
# =========================================================

if not policy_path.exists():
    raise FileNotFoundError(
        f"缺少Step34最终策略："
        f"{policy_path}"
    )

with policy_path.open(
    "r",
    encoding="utf-8",
) as file:
    policy = json.load(file)

required_policy_fields = [
    "router_threshold",
    "selector_threshold",
    "low_threshold",
    "high_threshold",
    "text_probability_threshold",
]

for field in required_policy_fields:
    if field not in policy:
        raise KeyError(
            f"策略缺少字段：{field}"
        )

router_threshold = float(
    policy["router_threshold"]
)

selector_threshold = float(
    policy["selector_threshold"]
)

low_threshold = float(
    policy["low_threshold"]
)

high_threshold = float(
    policy["high_threshold"]
)

text_probability_threshold = float(
    policy[
        "text_probability_threshold"
    ]
)

if low_threshold >= high_threshold:
    raise ValueError(
        "low_threshold必须小于high_threshold"
    )

print()
print("=" * 72)
print("Step35 最终比赛策略")
print("=" * 72)
print(
    "Router threshold：",
    router_threshold,
)
print(
    "Dual selector threshold：",
    selector_threshold,
)
print(
    "Speaker low：",
    low_threshold,
)
print(
    "Speaker high：",
    high_threshold,
)
print(
    "Text probability：",
    text_probability_threshold,
)

if "oof_metrics" in policy:
    print(
        "Step34 OOF指标：",
        policy["oof_metrics"],
    )


# =========================================================
# 7. 加载三个机器学习模型
# =========================================================

for model_path in [
    text_gate_model_path,
    router_model_path,
    selector_model_path,
]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"缺少模型：{model_path}"
        )


print(
    "\n加载最终文本门控……"
)

text_gate_bundle = joblib.load(
    text_gate_model_path
)

text_vectorizer = (
    text_gate_bundle[
        "vectorizer"
    ]
)

text_classifier = (
    text_gate_bundle[
        "classifier"
    ]
)

print(
    "最终文本门控加载完成"
)


print(
    "加载Second-pass Router……"
)

router_bundle = joblib.load(
    router_model_path
)

router_vectorizer = (
    router_bundle[
        "vectorizer"
    ]
)

router_classifier = (
    router_bundle[
        "classifier"
    ]
)

print(
    "Second-pass Router加载完成"
)


print(
    "加载Dual-ASR Selector……"
)

selector_bundle = joblib.load(
    selector_model_path
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

selector_classifier = (
    selector_bundle[
        "classifier"
    ]
)

print(
    "Dual-ASR Selector加载完成"
)


# =========================================================
# 8. 特征函数
# 必须与Step30 / Step33一致
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
        for i in range(
            1,
            len(text),
        )
    )

    return repeats / (
        len(text) - 1
    )


DOMAIN_WORDS = [
    "空调",
    "制冷",
    "制热",
    "除湿",
    "抽湿",
    "新风",
    "送风",
    "自动模式",
    "睡眠模式",
    "节能模式",
    "防直吹",
    "上下扫风",
    "左右扫风",
    "扫风",
    "风速",
    "风量",
    "风向",
    "出风口",
    "温度",
    "控温",
    "显示屏",
    "灯光",
    "亮度",
    "色温",
    "冷色调",
    "暖色调",
    "窗帘",
    "客厅",
    "厨房",
    "油烟机",
    "烟机",
    "滤网",
    "调高",
    "调低",
    "调大",
    "调小",
    "十六度",
    "十七度",
    "十八度",
    "十九度",
    "二十度",
    "二十一度",
    "二十二度",
    "二十三度",
    "二十四度",
    "二十五度",
    "二十六度",
    "二十七度",
    "二十八度",
    "二十九度",
    "三十度",
]


def domain_hits(text):
    return sum(
        word in text
        for word in DOMAIN_WORDS
    )


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


def overlap_ratio(a, b):
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    sa = set(a)
    sb = set(b)

    union = len(
        sa | sb
    )

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

    return n / max(
        len(a),
        len(b),
        1,
    )


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

    return n / max(
        len(a),
        len(b),
        1,
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


def positive_probability(
    classifier,
    features,
):
    classes = list(
        classifier.classes_
    )

    if 1 not in classes:
        raise RuntimeError(
            f"模型classes_中没有正类1："
            f"{classes}"
        )

    pos_col = classes.index(1)

    return float(
        classifier.predict_proba(
            features
        )[0, pos_col]
    )


def get_router_probability(
    para_text,
    speaker_score,
):
    text_features = (
        router_vectorizer.transform(
            [para_text]
        )
    )

    numeric = csr_matrix(
        np.asarray(
            [
                router_numeric_features(
                    para_text,
                    speaker_score,
                )
            ],
            dtype=np.float32,
        )
    )

    features = hstack(
        [
            text_features,
            numeric,
        ],
        format="csr",
    )

    return positive_probability(
        router_classifier,
        features,
    )


def get_selector_probability(
    para_text,
    sv_text,
    speaker_score,
):
    xp = selector_vec_para.transform(
        [para_text]
    )

    xs = selector_vec_sv.transform(
        [sv_text]
    )

    xn = csr_matrix(
        np.asarray(
            [
                selector_numeric_features(
                    para_text,
                    sv_text,
                    speaker_score,
                )
            ],
            dtype=np.float32,
        )
    )

    features = hstack(
        [
            xp,
            xs,
            xn,
        ],
        format="csr",
    )

    return positive_probability(
        selector_classifier,
        features,
    )


def get_text_probability(
    selected_text,
):
    features = (
        text_vectorizer.transform(
            [
                get_model_text(
                    selected_text
                )
            ]
        )
    )

    return positive_probability(
        text_classifier,
        features,
    )


# =========================================================
# 9. 加载CAM++ / Paraformer / SenseVoice
# =========================================================

print(
    "\n正在加载CAM++……"
)

speaker_model = pipeline(
    task="speaker-verification",
    model=(
        "iic/"
        "speech_campplus_sv_zh-cn_16k-common"
    ),
)

print(
    "CAM++加载完成"
)


asr_device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "\nASR运行设备：",
    asr_device,
)


print(
    "正在加载Paraformer……"
)

paraformer_model = AutoModel(
    model="paraformer-zh",
    device=asr_device,
    disable_update=True,
)

print(
    "Paraformer加载完成"
)


print(
    "正在加载SenseVoiceSmall……"
)

sensevoice_model = AutoModel(
    model="iic/SenseVoiceSmall",
    device=asr_device,
    disable_update=True,
)

print(
    "SenseVoiceSmall加载完成"
)


# =========================================================
# 10. 模型推理函数
# =========================================================

def extract_speaker_score(result):
    if not isinstance(
        result,
        dict,
    ):
        raise TypeError(
            f"CAM++返回类型异常："
            f"{type(result)}"
        )

    score = result.get(
        "score",
        0.0,
    )

    if isinstance(
        score,
        list,
    ):
        score = (
            score[0]
            if score
            else 0.0
        )

    if hasattr(
        score,
        "item",
    ):
        score = score.item()

    return float(score)


def get_speaker_score(
    wake_path,
    command_path,
):
    result = speaker_model(
        [
            str(wake_path),
            str(command_path),
        ]
    )

    return extract_speaker_score(
        result
    )


def run_paraformer(
    command_path,
):
    with torch.inference_mode():
        result = (
            paraformer_model.generate(
                input=str(
                    command_path
                ),
                hotword=(
                    hotword_string
                ),
            )
        )

    if (
        not result
        or not isinstance(
            result,
            list,
        )
    ):
        return ""

    first = result[0]

    if not isinstance(
        first,
        dict,
    ):
        return ""

    return normalize_text(
        first.get(
            "text",
            "",
        )
    )


def run_sensevoice(
    command_path,
):
    with torch.inference_mode():
        result = (
            sensevoice_model.generate(
                input=str(
                    command_path
                ),
                cache={},
                language="zh",
                use_itn=False,
            )
        )

    if (
        not result
        or not isinstance(
            result,
            list,
        )
    ):
        return ""

    first = result[0]

    if not isinstance(
        first,
        dict,
    ):
        return ""

    raw = first.get(
        "text",
        "",
    )

    try:
        raw = (
            rich_transcription_postprocess(
                raw
            )
        )
    except Exception:
        pass

    return normalize_text(
        raw,
        sensevoice=True,
    )


# =========================================================
# 11. 最终推理
# =========================================================

results = []
detail_rows = []

speaker_call_count = 0
paraformer_call_count = 0
router_call_count = 0
sensevoice_call_count = 0
selector_call_count = 0
text_gate_call_count = 0

low_reject_count = 0
high_accept_count = 0
middle_accept_count = 0
middle_reject_count = 0

selected_para_only_count = 0
selected_para_after_sv_count = 0
selected_sv_count = 0

start_time = time.perf_counter()


for sample in tqdm(
    samples,
    desc="Step37最终选择性双ASR推理",
):
    sample_start = (
        time.perf_counter()
    )

    # -----------------------------------------------------
    # Speaker
    # -----------------------------------------------------

    speaker_start = (
        time.perf_counter()
    )

    speaker_score = (
        get_speaker_score(
            sample["wake_path"],
            sample[
                "command_path"
            ],
        )
    )

    speaker_ms = (
        time.perf_counter()
        - speaker_start
    ) * 1000

    speaker_call_count += 1


    para_text = ""
    sv_text = ""
    selected_text = ""
    final_text = ""

    router_probability = None
    selector_probability = None
    text_probability = None

    para_ms = 0.0
    router_ms = 0.0
    sv_ms = 0.0
    selector_ms = 0.0
    text_gate_ms = 0.0

    selected_model = "none"

    # -----------------------------------------------------
    # Low reject
    # -----------------------------------------------------

    if speaker_score < low_threshold:

        decision = (
            "低分直接拒绝"
        )

        low_reject_count += 1

    else:
        # -------------------------------------------------
        # Paraformer first pass
        # -------------------------------------------------

        para_start = (
            time.perf_counter()
        )

        para_text = run_paraformer(
            sample["command_path"]
        )

        para_ms = (
            time.perf_counter()
            - para_start
        ) * 1000

        paraformer_call_count += 1

        selected_text = para_text
        selected_model = (
            "paraformer_only"
        )

        # -------------------------------------------------
        # Second-pass router
        # -------------------------------------------------

        router_start = (
            time.perf_counter()
        )

        router_probability = (
            get_router_probability(
                para_text,
                speaker_score,
            )
        )

        router_ms = (
            time.perf_counter()
            - router_start
        ) * 1000

        router_call_count += 1

        if (
            router_probability
            >= router_threshold
        ):
            # ---------------------------------------------
            # SenseVoice second pass
            # ---------------------------------------------

            sv_start = (
                time.perf_counter()
            )

            sv_text = run_sensevoice(
                sample[
                    "command_path"
                ]
            )

            sv_ms = (
                time.perf_counter()
                - sv_start
            ) * 1000

            sensevoice_call_count += 1

            # ---------------------------------------------
            # Dual-ASR selector
            # ---------------------------------------------

            selector_start = (
                time.perf_counter()
            )

            selector_probability = (
                get_selector_probability(
                    para_text,
                    sv_text,
                    speaker_score,
                )
            )

            selector_ms = (
                time.perf_counter()
                - selector_start
            ) * 1000

            selector_call_count += 1

            if (
                selector_probability
                >= selector_threshold
            ):
                selected_text = sv_text
                selected_model = (
                    "sensevoice"
                )
                selected_sv_count += 1

            else:
                selected_text = (
                    para_text
                )
                selected_model = (
                    "paraformer_after_sv"
                )
                selected_para_after_sv_count += 1

        else:
            selected_para_only_count += 1

        # -------------------------------------------------
        # Speaker high / text gate
        # 与Step34评估逻辑一致：
        # ASR与Router先执行，再决定最终是否接受。
        # -------------------------------------------------

        if (
            speaker_score
            >= high_threshold
        ):
            final_text = (
                selected_text
            )

            decision = (
                "高分直接接受"
            )

            high_accept_count += 1

        else:
            gate_start = (
                time.perf_counter()
            )

            text_probability = (
                get_text_probability(
                    selected_text
                )
            )

            text_gate_ms = (
                time.perf_counter()
                - gate_start
            ) * 1000

            text_gate_call_count += 1

            if (
                text_probability
                >= text_probability_threshold
            ):
                final_text = (
                    selected_text
                )

                decision = (
                    "文本门控接受"
                )

                middle_accept_count += 1

            else:
                final_text = ""

                decision = (
                    "文本门控拒绝"
                )

                middle_reject_count += 1


    sample_ms = (
        time.perf_counter()
        - sample_start
    ) * 1000


    # -----------------------------------------------------
    # 提交结果
    # -----------------------------------------------------

    result_item = {
        "id": sample[
            "sample_id"
        ],
        "content": final_text,
    }

    if (
        INCLUDE_LOCAL_METRICS_WHEN_AVAILABLE
        and sample[
            "label_available"
        ]
    ):
        reference = (
            sample["reference"]
        )

        result_item[
            "label"
        ] = reference

        if reference:
            result_item[
                "cer"
            ] = (
                editdistance.eval(
                    reference,
                    final_text,
                )
                / len(reference)
            )
        else:
            result_item[
                "cer"
            ] = None

    results.append(
        result_item
    )


    detail_rows.append(
        {
            "source_file": (
                sample[
                    "source_file"
                ]
            ),
            "source_line": (
                sample[
                    "source_line"
                ]
            ),
            "id": (
                sample[
                    "sample_id"
                ]
            ),
            "wake_path": str(
                sample[
                    "wake_path"
                ]
            ),
            "command_path": str(
                sample[
                    "command_path"
                ]
            ),
            "reference": (
                sample[
                    "reference"
                ]
            ),
            "speaker_score": (
                speaker_score
            ),
            "low_threshold": (
                low_threshold
            ),
            "high_threshold": (
                high_threshold
            ),
            "router_probability": (
                ""
                if router_probability is None
                else router_probability
            ),
            "router_threshold": (
                router_threshold
            ),
            "paraformer_text": (
                para_text
            ),
            "sensevoice_text": (
                sv_text
            ),
            "selector_probability": (
                ""
                if selector_probability is None
                else selector_probability
            ),
            "selector_threshold": (
                selector_threshold
            ),
            "selected_model": (
                selected_model
            ),
            "selected_text": (
                selected_text
            ),
            "text_probability": (
                ""
                if text_probability is None
                else text_probability
            ),
            "text_probability_threshold": (
                text_probability_threshold
            ),
            "decision": (
                decision
            ),
            "final_text": (
                final_text
            ),
            "speaker_duration_ms": (
                speaker_ms
            ),
            "paraformer_duration_ms": (
                para_ms
            ),
            "router_duration_ms": (
                router_ms
            ),
            "sensevoice_duration_ms": (
                sv_ms
            ),
            "selector_duration_ms": (
                selector_ms
            ),
            "text_gate_duration_ms": (
                text_gate_ms
            ),
            "sample_duration_ms": (
                sample_ms
            ),
        }
    )


total_duration_seconds = (
    time.perf_counter()
    - start_time
)

submission_duration_seconds = round(
    total_duration_seconds,
    3,
)


# =========================================================
# 12. 本地TestA指标
# =========================================================

total_target_chars = 0
total_target_distance = 0

neg_count = 0
neg_reject_count = 0

for sample, result_item in zip(
    samples,
    results,
):
    if not sample[
        "label_available"
    ]:
        continue

    reference = (
        sample["reference"]
    )

    hypothesis = (
        result_item[
            "content"
        ]
    )

    if reference:
        total_target_chars += (
            len(reference)
        )

        total_target_distance += (
            editdistance.eval(
                reference,
                hypothesis,
            )
        )

    else:
        neg_count += 1

        if hypothesis == "":
            neg_reject_count += 1


final_cer = (
    total_target_distance
    / total_target_chars
    if total_target_chars
    else None
)

rr = (
    neg_reject_count
    / neg_count
    if neg_count
    else None
)

recognition_score = (
    0.5 * (1.0 - final_cer)
    + 0.5 * rr
    if (
        final_cer is not None
        and rr is not None
    )
    else None
)


# =========================================================
# 13. 保存结果JSON
# =========================================================

result_payload = {
    "results": results,
    "duration": (
        submission_duration_seconds
    ),
}

if (
    INCLUDE_LOCAL_METRICS_WHEN_AVAILABLE
    and final_cer is not None
):
    result_payload[
        "final_cer"
    ] = final_cer

if (
    INCLUDE_LOCAL_METRICS_WHEN_AVAILABLE
    and (not SUBMISSION_MODE)
    and rr is not None
):
    result_payload[
        "local_rr"
    ] = rr

if (
    INCLUDE_LOCAL_METRICS_WHEN_AVAILABLE
    and (not SUBMISSION_MODE)
    and recognition_score is not None
):
    result_payload[
        "local_recognition_score"
    ] = recognition_score

result_data = {
    "result": result_payload
}

with result_json_path.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        result_data,
        file,
        ensure_ascii=False,
        indent=2,
    )


# =========================================================
# 14. 保存详情CSV
# =========================================================

with detail_csv_path.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:

    fieldnames = list(
        detail_rows[0].keys()
    ) if detail_rows else []

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        detail_rows
    )


# =========================================================
# 15. 完整性 / 汇总
# =========================================================

if len(results) != len(samples):
    raise RuntimeError(
        "输入输出数量不一致"
    )

if (
    len(
        {
            item["id"]
            for item in results
        }
    )
    != len(results)
):
    raise RuntimeError(
        "最终结果存在重复ID"
    )


sample_count = len(samples)

paraformer_call_rate = (
    paraformer_call_count
    / sample_count
    if sample_count
    else 0.0
)

sensevoice_call_rate = (
    sensevoice_call_count
    / sample_count
    if sample_count
    else 0.0
)

asr_model_calls_per_sample = (
    (
        paraformer_call_count
        + sensevoice_call_count
    )
    / sample_count
    if sample_count
    else 0.0
)

router_call_rate = (
    router_call_count
    / sample_count
    if sample_count
    else 0.0
)

text_gate_call_rate = (
    text_gate_call_count
    / sample_count
    if sample_count
    else 0.0
)

empty_output_rate = (
    sum(
        item["content"] == ""
        for item in results
    )
    / sample_count
    if sample_count
    else 0.0
)



# =========================================================
# 15.5 提交JSON结构自检
# =========================================================

loaded_submission = json.loads(
    result_json_path.read_text(
        encoding="utf-8"
    )
)

if "result" not in loaded_submission:
    raise RuntimeError(
        "结果JSON缺少顶层 result"
    )

submission_result = loaded_submission["result"]

if "results" not in submission_result:
    raise RuntimeError(
        "结果JSON缺少 result.results"
    )

if "duration" not in submission_result:
    raise RuntimeError(
        "结果JSON缺少 result.duration"
    )

if len(submission_result["results"]) != sample_count:
    raise RuntimeError(
        "结果JSON中的results数量与输入不一致"
    )

for i, item in enumerate(
    submission_result["results"]
):
    if "id" not in item:
        raise RuntimeError(
            f"第{i}条结果缺少id"
        )

    if "content" not in item:
        raise RuntimeError(
            f"第{i}条结果缺少content"
        )

if len({
    item["id"]
    for item in submission_result["results"]
}) != sample_count:
    raise RuntimeError(
        "结果JSON中的id不唯一"
    )

print()
print("=" * 80)
print("Step37 最终选择性双ASR推理完成")
print("=" * 80)

print(
    "样本数：",
    sample_count,
)

print(
    "Paraformer调用率：",
    paraformer_call_rate,
)

print(
    "SenseVoice调用率：",
    sensevoice_call_rate,
)

print(
    "平均ASR模型调用次数/样本：",
    asr_model_calls_per_sample,
)

print(
    "Router调用率：",
    router_call_rate,
)

print(
    "文本门控调用率：",
    text_gate_call_rate,
)

print(
    "选Paraformer-only：",
    selected_para_only_count,
)

print(
    "跑SV后仍选Paraformer：",
    selected_para_after_sv_count,
)

print(
    "最终选SenseVoice：",
    selected_sv_count,
)

print(
    "低分直接拒绝：",
    low_reject_count,
)

print(
    "高分直接接受：",
    high_accept_count,
)

print(
    "文本门控接受：",
    middle_accept_count,
)

print(
    "文本门控拒绝：",
    middle_reject_count,
)

print(
    "空输出比例：",
    empty_output_rate,
)

print(
    "总推理时间(s)：",
    submission_duration_seconds,
)

if final_cer is not None:
    print(
        "TestA目标发音人CER：",
        final_cer,
        f"({final_cer * 100:.3f}%)",
    )

if rr is not None:
    print(
        "TestA拒识率RR：",
        rr,
        f"({rr * 100:.3f}%)",
    )

if recognition_score is not None:
    print(
        "本地识别平衡分：",
        recognition_score,
    )

print(
    "结果文件：",
    result_json_path,
)

print(
    "详细日志：",
    detail_csv_path,
)

print()
print("JSON结构自检：通过")
print()
print("TestA本地运行示例：")
print(
    r'python step37_competition_inference.py '
    r'--data-dir data/testA'
)
print()
print("正式提交/隐藏B示例：")
print(
    r'python step37_competition_inference.py '
    r'--data-dir data/testB --submission '
    r'--output-json output/final_testB_result.json'
)