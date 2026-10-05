# ============================================================
# 桌球自動發球機精準落點與四參數智慧控制模型
# 完整版本
#
# 第一階段：
# 四個參數 → 落點預測
#
# 第二階段：
# 目標落點 / 目標範圍
# → 四參數搜尋
# → 落點準確優先
# → 球路條件次要最佳化
# → 產生多組同落點、不同球路方案
# ============================================================


# ============================================================
# 第 1 格：載入套件
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import files

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from sklearn.model_selection import GroupShuffleSplit

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import joblib
import re
import heapq
from itertools import product

print("✅ 套件載入完成")


# ============================================================
# 第 2 格：上傳第一階段 Excel
# ============================================================

uploaded = files.upload()

if len(uploaded) == 0:
    raise ValueError("❌ 沒有上傳檔案")

filename = list(uploaded.keys())[0]

print("✅ 已上傳：", filename)

excel_file = pd.ExcelFile(filename)

print("\nExcel 工作表：")
print(excel_file.sheet_names)


# ============================================================
# 第 3 格：讀取第一階段資料
# ============================================================

target_sheet = "第一階段1000組"

if target_sheet not in excel_file.sheet_names:
    raise ValueError(
        f"找不到工作表「{target_sheet}」，\n"
        f"目前工作表為：{excel_file.sheet_names}"
    )

df = pd.read_excel(
    filename,
    sheet_name=target_sheet
)

# 清理欄位名稱
df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

print("✅ 資料讀取完成")
print("資料大小：", df.shape)

print("\n欄位名稱：")
print(df.columns.tolist())


# ============================================================
# 第 4 格：指定模型欄位
# ============================================================

feature_columns = [
    "仰角角度",
    "左右位置",
    "旋球角度",
    "出球強度"
]

shot_columns = [
    "發球1_落點",
    "發球2_落點",
    "發球3_落點",
    "發球4_落點",
    "發球5_落點"
]

id_column = "測試編號"

required_columns = (
    [id_column]
    + feature_columns
    + shot_columns
)

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "❌ 缺少以下欄位：\n"
        + "\n".join(missing_columns)
    )

print("✅ 所有需要的欄位都存在")


# ============================================================
# 第 5 格：四個參數合法性檢查
# ============================================================

parameter_limits = {
    "仰角角度": (1, 28),
    "左右位置": (1, 9),
    "旋球角度": (1, 59),
    "出球強度": (1, 99)
}

for col, (min_value, max_value) in parameter_limits.items():

    # 轉數字
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    # 空值
    missing_count = df[col].isna().sum()

    # 超出設備範圍
    out_of_range = (
        (df[col] < min_value)
        |
        (df[col] > max_value)
    ).sum()

    # 非整數
    non_integer_count = (
        ~df[col].isna()
        &
        (df[col] % 1 != 0)
    ).sum()

    print(f"\n【{col}】")
    print("最小值 =", df[col].min())
    print("最大值 =", df[col].max())
    print("空白 =", missing_count)
    print("超出範圍 =", out_of_range)
    print("非整數 =", non_integer_count)

    if missing_count > 0:
        raise ValueError(
            f"❌ {col} 有空白資料"
        )

    if out_of_range > 0:
        raise ValueError(
            f"❌ {col} 有超出設備範圍的資料"
        )

    if non_integer_count > 0:
        raise ValueError(
            f"❌ {col} 有非整數資料"
        )

    df[col] = df[col].astype(int)

print("\n✅ 四個參數全部通過檢查")


# ============================================================
# 第 6 格：建立落點類別
# ============================================================

VALID_LABELS = [
    f"({x},{y})"
    for x in range(1, 6)
    for y in range(1, 6)
]

# 4 種無效
INVALID_LABELS = [
    "飛出台面",
    "未過網",
    "其他無效",
    "球碰網後沒有進入有效區"
]

# 總共 25 + 4 = 29 類
ALL_LABELS = (
    VALID_LABELS
    +
    INVALID_LABELS
)

print("有效落點數量：", len(VALID_LABELS))
print("無效類別數量：", len(INVALID_LABELS))
print("總類別數量：", len(ALL_LABELS))


# ============================================================
# 第 7 格：落點文字整理
# ============================================================

def clean_landing_point(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # 全形符號轉半形
    value = (
        value
        .replace("（", "(")
        .replace("）", ")")
        .replace("，", ",")
    )

    # 移除空白
    value = re.sub(
        r"\s+",
        "",
        value
    )

    # --------------------------------------------------------
    # 使用者確認：
    # 碰桌後彈出 = 其他無效
    # --------------------------------------------------------

    if value == "碰桌後彈出":
        return "其他無效"

    # --------------------------------------------------------
    # 四種無效
    # --------------------------------------------------------

    if value in INVALID_LABELS:
        return value

    # --------------------------------------------------------
    # 座標格式：
    #
    # (4,3)
    # 4,3
    # （4，3）
    # --------------------------------------------------------

    match = re.fullmatch(
        r"\(([1-5]),([1-5])\)|([1-5]),([1-5])",
        value
    )

    if match:

        if match.group(1) is not None:
            x = match.group(1)
            y = match.group(2)
        else:
            x = match.group(3)
            y = match.group(4)

        return f"({x},{y})"

    # --------------------------------------------------------
    # 無法辨識
    # --------------------------------------------------------

    return np.nan


# 整理 5 次發球
for col in shot_columns:

    df[col] = (
        df[col]
        .apply(clean_landing_point)
    )

print("✅ 落點整理完成")


# ============================================================
# 第 8 格：寬資料 → 長資料
# ============================================================

long_df = df.melt(
    id_vars=[
        id_column,
        *feature_columns
    ],
    value_vars=shot_columns,
    var_name="發球次數",
    value_name="落點"
)

print("✅ 5 次發球已拆開")
print("資料筆數：", len(long_df))

display(
    long_df.head(15)
)


# ============================================================
# 第 9 格：檢查空白落點
# ============================================================

blank_mask = long_df["落點"].isna()

blank_count = blank_mask.sum()

print(
    "空白落點數量：",
    blank_count
)

if blank_count > 0:

    print("\n以下資料沒有落點：")

    display(
        long_df.loc[
            blank_mask,
            [
                id_column,
                *feature_columns,
                "發球次數",
                "落點"
            ]
        ]
    )

    print(
        "\n⚠️ 沒有實測結果的球不會拿來訓練。"
    )

# 移除沒有結果的球
long_df = (
    long_df
    .dropna(
        subset=["落點"]
    )
    .copy()
)

print("\n✅ 清理完成")
print("剩餘可用球數：", len(long_df))


# ============================================================
# 第 10 格：檢查 29 類結果
# ============================================================

actual_labels = sorted(
    long_df["落點"]
    .unique()
    .tolist()
)

print(
    "實際出現的類別數量：",
    len(actual_labels)
)

print("\n實際類別：")

for label in actual_labels:
    print(label)

print(
    "\n預期總類別數量：",
    len(ALL_LABELS)
)

unknown_labels = [
    label
    for label in actual_labels
    if label not in ALL_LABELS
]

if unknown_labels:

    print("\n❌ 發現未知標籤：")

    for label in unknown_labels:
        print(label)

    raise ValueError(
        "請檢查 Excel 中的落點文字格式。"
    )

print("\n✅ 所有結果都符合 29 類設定")


# ============================================================
# 第 11 格：查看落點分布
# ============================================================

class_counts = (
    long_df["落點"]
    .value_counts()
    .reindex(
        ALL_LABELS,
        fill_value=0
    )
)

distribution_df = pd.DataFrame({
    "落點": class_counts.index,
    "球數": class_counts.values
})

distribution_df["比例"] = (
    distribution_df["球數"]
    /
    distribution_df["球數"].sum()
    *
    100
).round(2)

display(
    distribution_df
)

print(
    "\n總球數：",
    distribution_df["球數"].sum()
)


# ============================================================
# 第 12 格：建立 X / y / groups
# ============================================================

X = long_df[
    feature_columns
].copy()

y = long_df[
    "落點"
].copy()

groups = long_df[
    id_column
].copy()

print("X 大小：", X.shape)
print("y 大小：", y.shape)
print(
    "Group 數量：",
    groups.nunique()
)

print("\nX：")
display(
    X.head()
)

print("\ny：")
display(
    y.head()
)


# ============================================================
# 第 13 格：Group Train / Test Split
# ============================================================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[
    train_idx
].copy()

X_test = X.iloc[
    test_idx
].copy()

y_train = y.iloc[
    train_idx
].copy()

y_test = y.iloc[
    test_idx
].copy()

groups_train = groups.iloc[
    train_idx
]

groups_test = groups.iloc[
    test_idx
]

print("========== Training ==========")
print(
    "球數：",
    len(X_train)
)

print(
    "參數組數：",
    groups_train.nunique()
)

print("\n========== Testing ==========")
print(
    "球數：",
    len(X_test)
)

print(
    "參數組數：",
    groups_test.nunique()
)

# ------------------------------------------------------------
# 確認沒有 Group 重疊
# ------------------------------------------------------------

overlap = (
    set(groups_train.unique())
    &
    set(groups_test.unique())
)

print(
    "\nTrain/Test Group 重疊數量：",
    len(overlap)
)

if len(overlap) > 0:

    raise ValueError(
        "❌ 發生資料洩漏：Train/Test 有相同測試編號！"
    )

print(
    "✅ Train/Test 沒有 Group 重疊"
)


# ============================================================
# 第 14 格：Random Forest
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

print(
    "✅ Random Forest 訓練完成"
)


# ============================================================
# 第 15 格：Extra Trees
# ============================================================

et_model = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

et_model.fit(
    X_train,
    y_train
)

print(
    "✅ Extra Trees 訓練完成"
)


# ============================================================
# 第 16 格：模型比較
# ============================================================

rf_pred = rf_model.predict(
    X_test
)

et_pred = et_model.predict(
    X_test
)

results = []

for model_name, pred in [
    ("Random Forest", rf_pred),
    ("Extra Trees", et_pred)
]:

    results.append({

        "模型": model_name,

        "Accuracy":
            accuracy_score(
                y_test,
                pred
            ),

        "Balanced Accuracy":
            balanced_accuracy_score(
                y_test,
                pred
            ),

        "Macro F1":
            f1_score(
                y_test,
                pred,
                average="macro"
            ),

        "Weighted F1":
            f1_score(
                y_test,
                pred,
                average="weighted"
            )
    })

model_comparison = (
    pd.DataFrame(results)
)

display(
    model_comparison.style.format({
        "Accuracy": "{:.4f}",
        "Balanced Accuracy": "{:.4f}",
        "Macro F1": "{:.4f}",
        "Weighted F1": "{:.4f}"
    })
)


# ============================================================
# 第 17 格：選擇最佳模型
# ============================================================

best_row = model_comparison.loc[
    model_comparison["Macro F1"].idxmax()
]

best_model_name = (
    best_row["模型"]
)

if best_model_name == "Random Forest":

    best_model = rf_model

    best_pred = rf_pred

else:

    best_model = et_model

    best_pred = et_pred

print("================================")
print(
    "最佳模型：",
    best_model_name
)
print("================================")

print(
    f"Accuracy          = "
    f"{best_row['Accuracy']:.4f}"
)

print(
    f"Balanced Accuracy = "
    f"{best_row['Balanced Accuracy']:.4f}"
)

print(
    f"Macro F1          = "
    f"{best_row['Macro F1']:.4f}"
)

print(
    f"Weighted F1       = "
    f"{best_row['Weighted F1']:.4f}"
)


# ============================================================
# 第 18 格：Classification Report
# ============================================================

print(
    classification_report(
        y_test,
        best_pred,
        labels=ALL_LABELS,
        zero_division=0
    )
)


# ============================================================
# 第 19 格：Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    best_pred,
    labels=ALL_LABELS
)

plt.figure(
    figsize=(14, 12)
)

plt.imshow(cm)

plt.xticks(
    range(len(ALL_LABELS)),
    ALL_LABELS,
    rotation=90
)

plt.yticks(
    range(len(ALL_LABELS)),
    ALL_LABELS
)

plt.xlabel(
    "模型預測"
)

plt.ylabel(
    "實際結果"
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.colorbar()

plt.tight_layout()

plt.show()


# ============================================================
# 第 20 格：落點空間距離
# ============================================================

def parse_coordinate(label):

    if not isinstance(
        label,
        str
    ):
        return None

    match = re.fullmatch(
        r"\(([1-5]),([1-5])\)",
        label
    )

    if not match:
        return None

    return (
        int(match.group(1)),
        int(match.group(2))
    )


spatial_records = []

for actual, pred in zip(
    y_test,
    best_pred
):

    actual_xy = parse_coordinate(
        actual
    )

    pred_xy = parse_coordinate(
        pred
    )

    if (
        actual_xy is not None
        and
        pred_xy is not None
    ):

        dx = (
            actual_xy[0]
            -
            pred_xy[0]
        )

        dy = (
            actual_xy[1]
            -
            pred_xy[1]
        )

        manhattan = (
            abs(dx)
            +
            abs(dy)
        )

        euclidean = np.sqrt(
            dx**2 + dy**2
        )

        spatial_records.append({

            "實際":
                actual,

            "預測":
                pred,

            "Manhattan":
                manhattan,

            "Euclidean":
                euclidean
        })


spatial_df = pd.DataFrame(
    spatial_records
)

if len(spatial_df) > 0:

    print(
        "有效座標預測筆數：",
        len(spatial_df)
    )

    print(
        "平均 Manhattan Distance：",
        round(
            spatial_df["Manhattan"].mean(),
            4
        )
    )

    print(
        "平均 Euclidean Distance：",
        round(
            spatial_df["Euclidean"].mean(),
            4
        )
    )

    print(
        "預測完全正確比例：",
        round(
            (
                spatial_df["Manhattan"]
                ==
                0
            ).mean(),
            4
        )
    )

    print(
        "預測差 1 格以內比例：",
        round(
            (
                spatial_df["Manhattan"]
                <=
                1
            ).mean(),
            4
        )
    )

else:

    print(
        "⚠️ 沒有足夠的有效座標資料可以計算空間距離。"
    )


# ============================================================
# 第 21 格：Feature Importance
# ============================================================

importance_df = pd.DataFrame({

    "參數":
        feature_columns,

    "重要程度":
        best_model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        "重要程度",
        ascending=False
    )
    .reset_index(drop=True)
)

display(
    importance_df
)


# ============================================================
# 第 22 格：建立一些共用函數
# ============================================================

# ------------------------------------------------------------
# 模型類別 → index
# ------------------------------------------------------------

class_to_index = {
    cls: i
    for i, cls
    in enumerate(
        best_model.classes_
    )
}

print(
    "模型目前實際學到的類別數：",
    len(best_model.classes_)
)

missing_model_classes = [
    label
    for label in ALL_LABELS
    if label not in class_to_index
]

if missing_model_classes:

    print(
        "\n⚠️ 下列類別在 Training 中沒有出現，"
        "因此模型目前無法直接預測這些類別："
    )

    for label in missing_model_classes:
        print(label)

else:

    print(
        "✅ 29 類全部存在於模型中"
    )


# ------------------------------------------------------------
# 無效機率
# ------------------------------------------------------------

def calculate_invalid_probability(
    probabilities
):

    invalid_prob = np.zeros(
        len(probabilities)
    )

    for invalid_label in INVALID_LABELS:

        if invalid_label in class_to_index:

            invalid_index = (
                class_to_index[
                    invalid_label
                ]
            )

            invalid_prob += (
                probabilities[
                    :,
                    invalid_index
                ]
            )

    return invalid_prob


# ------------------------------------------------------------
# 目標標籤整理
# ------------------------------------------------------------

def normalize_target_labels(
    target_landing
):

    # 單一座標
    if isinstance(
        target_landing,
        str
    ):

        labels = [
            target_landing
        ]

    # 多個座標
    elif isinstance(
        target_landing,
        (list, tuple, set)
    ):

        labels = list(
            target_landing
        )

    else:

        raise ValueError(
            "target_landing 必須是 "
            "例如 '(4,3)' 或 ['(4,3)', '(4,4)']"
        )

    normalized = []

    for label in labels:

        cleaned = clean_landing_point(
            label
        )

        if (
            pd.isna(cleaned)
            or
            cleaned not in VALID_LABELS
        ):

            raise ValueError(
                f"無效目標落點：{label}\n"
                f"必須是 (1,1)~(5,5)"
            )

        normalized.append(
            cleaned
        )

    return list(
        dict.fromkeys(normalized)
    )


# ------------------------------------------------------------
# 找座標到目標範圍的最小 Manhattan Distance
# ------------------------------------------------------------

def distance_to_targets(
    predicted_label,
    target_labels
):

    pred_xy = parse_coordinate(
        predicted_label
    )

    if pred_xy is None:
        return np.inf

    distances = []

    for target in target_labels:

        target_xy = parse_coordinate(
            target
        )

        if target_xy is None:
            continue

        dx = (
            pred_xy[0]
            -
            target_xy[0]
        )

        dy = (
            pred_xy[1]
            -
            target_xy[1]
        )

        distances.append(
            abs(dx) + abs(dy)
        )

    if not distances:
        return np.inf

    return min(distances)


# ------------------------------------------------------------
# 球路條件
#
# 注意：
# 目前資料只有「旋球角度」，
# 所以「高旋轉」實際上是指
# 「旋球角度偏高」。
# ------------------------------------------------------------

BALL_CONDITION_MAP = {

    "標準": [],

    "一般": [],

    "高速球": [
        ("出球強度", "max")
    ],

    "高強度": [
        ("出球強度", "max")
    ],

    "低速球": [
        ("出球強度", "min")
    ],

    "低強度": [
        ("出球強度", "min")
    ],

    "高旋轉": [
        ("旋球角度", "max")
    ],

    "高旋球角度": [
        ("旋球角度", "max")
    ],

    "低旋轉": [
        ("旋球角度", "min")
    ],

    "低旋球角度": [
        ("旋球角度", "min")
    ],

    "高仰角": [
        ("仰角角度", "max")
    ],

    "低仰角": [
        ("仰角角度", "min")
    ],

    "偏右": [
        ("左右位置", "max")
    ],

    "偏左": [
        ("左右位置", "min")
    ]
}


def normalize_ball_conditions(
    ball_condition
):

    if ball_condition is None:
        return ["標準"]

    if isinstance(
        ball_condition,
        str
    ):
        conditions = [
            ball_condition
        ]

    elif isinstance(
        ball_condition,
        (list, tuple, set)
    ):
        conditions = list(
            ball_condition
        )

    else:

        raise ValueError(
            "ball_condition 必須是字串或 list"
        )

    for condition in conditions:

        if condition not in BALL_CONDITION_MAP:

            raise ValueError(
                f"未知球路條件：{condition}\n"
                f"可使用："
                f"{list(BALL_CONDITION_MAP.keys())}"
            )

    return list(
        dict.fromkeys(conditions)
    )


# ------------------------------------------------------------
# 將參數轉成 0~1
# ------------------------------------------------------------

def normalized_parameter_series(
    df_input
):

    result = pd.DataFrame(
        index=df_input.index
    )

    for col, (
        min_value,
        max_value
    ) in parameter_limits.items():

        result[col] = (
            df_input[col]
            -
            min_value
        ) / (
            max_value
            -
            min_value
        )

    return result.clip(
        0,
        1
    )


# ------------------------------------------------------------
# 計算球路條件分數
#
# 0 = 完全不符合
# 1 = 最符合
# ------------------------------------------------------------

def calculate_condition_score(
    candidate_df,
    ball_condition
):

    conditions = normalize_ball_conditions(
        ball_condition
    )

    norm_df = (
        normalized_parameter_series(
            candidate_df
        )
    )

    score_list = []

    for condition in conditions:

        rules = BALL_CONDITION_MAP[
            condition
        ]

        # 標準 / 一般
        if len(rules) == 0:

            score = pd.Series(
                0.5,
                index=candidate_df.index
            )

        else:

            rule_scores = []

            for col, direction in rules:

                value = norm_df[col]

                if direction == "max":
                    rule_score = value

                elif direction == "min":
                    rule_score = 1 - value

                else:
                    raise ValueError(
                        f"未知方向：{direction}"
                    )

                rule_scores.append(
                    rule_score
                )

            score = sum(
                rule_scores
            ) / len(
                rule_scores
            )

        score_list.append(
            score
        )

    final_score = sum(
        score_list
    ) / len(
        score_list
    )

    return final_score


print(
    "✅ 共用函數建立完成"
)


# ============================================================
# 第 23 格：單組參數預測
# ============================================================

def validate_parameter_values(
    elevation,
    horizontal,
    spin_angle,
    power
):

    values = {

        "仰角角度":
            elevation,

        "左右位置":
            horizontal,

        "旋球角度":
            spin_angle,

        "出球強度":
            power
    }

    for col, value in values.items():

        min_value, max_value = (
            parameter_limits[col]
        )

        if pd.isna(value):

            raise ValueError(
                f"{col} 不可以是空值"
            )

        if (
            value < min_value
            or
            value > max_value
        ):

            raise ValueError(
                f"{col} 必須在 "
                f"{min_value}~{max_value} 之間"
            )

        if int(value) != value:

            raise ValueError(
                f"{col} 必須是整數"
            )

    return {
        "仰角角度":
            int(elevation),

        "左右位置":
            int(horizontal),

        "旋球角度":
            int(spin_angle),

        "出球強度":
            int(power)
    }


def predict_landing(
    elevation,
    horizontal,
    spin_angle,
    power,
    top_n=10
):

    values = validate_parameter_values(
        elevation,
        horizontal,
        spin_angle,
        power
    )

    input_df = pd.DataFrame(
        [values]
    )

    prediction = (
        best_model
        .predict(input_df)[0]
    )

    probabilities = (
        best_model
        .predict_proba(input_df)[0]
    )

    probability_df = pd.DataFrame({

        "落點":
            best_model.classes_,

        "預測機率":
            probabilities
    })

    probability_df = (
        probability_df
        .sort_values(
            "預測機率",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return (
        prediction,
        probability_df.head(top_n)
    )


# ------------------------------------------------------------
# 範例
# ------------------------------------------------------------

prediction, probability_df = (
    predict_landing(
        elevation=18,
        horizontal=5,
        spin_angle=30,
        power=65,
        top_n=10
    )
)

print(
    "\n模型預測結果：",
    prediction
)

print(
    "\n前 10 個最可能結果："
)

display(
    probability_df
)


# ============================================================
# 第 24 格：第一階段已測參數 Set
# ============================================================

tested_parameters = set(
    zip(
        df["仰角角度"],
        df["左右位置"],
        df["旋球角度"],
        df["出球強度"]
    )
)

print(
    "第一階段已測參數組合：",
    len(tested_parameters)
)


# ============================================================
# 第 25 格：
# 目標落點 → 搜尋四參數
#
# 重要排序原則：
#
# 第一優先：
#     模型預測結果有沒有落在目標
#
# 第二優先：
#     目標落點機率
#
# 第三優先：
#     無效球機率
#
# 第四優先：
#     距離目標多遠
#
# 第五優先：
#     球路條件分數
# ============================================================

def rank_candidate_dataframe(
    candidate_df
):

    return (
        candidate_df
        .sort_values(
            by=[
                "命中目標",
                "目標落點機率",
                "無效總機率",
                "目標距離_曼哈頓",
                "球路條件分數"
            ],
            ascending=[
                False,
                False,
                True,
                True,
                False
            ]
        )
        .reset_index(drop=True)
    )


def search_best_parameters(
    target_landing,
    ball_condition="標準",
    top_n=20,
    chunk_size=50000,
    pool_size=None
):

    target_labels = (
        normalize_target_labels(
            target_landing
        )
    )

    conditions = (
        normalize_ball_conditions(
            ball_condition
        )
    )

    print("==========================================")
    print("開始搜尋最佳四參數")
    print("==========================================")

    print(
        "目標落點 / 範圍：",
        target_labels
    )

    print(
        "球路條件：",
        conditions
    )

    # --------------------------------------------------------
    # 檢查目標類別是否存在於模型
    # --------------------------------------------------------

    missing_targets = [
        label
        for label in target_labels
        if label not in class_to_index
    ]

    if missing_targets:

        raise ValueError(
            "目前模型沒有學到目標類別：\n"
            +
            "\n".join(missing_targets)
            +
            "\n\n請確認 Training 中有這些落點資料。"
        )

    # --------------------------------------------------------
    # 完整搜尋空間
    # --------------------------------------------------------

    elevations = range(1, 29)
    horizontals = range(1, 10)
    spins = range(1, 60)
    powers = range(1, 100)

    total_combinations = (
        28
        *
        9
        *
        59
        *
        99
    )

    print(
        "\n所有可能參數組合：",
        f"{total_combinations:,}"
    )

    print(
        "第一階段已測：",
        f"{len(tested_parameters):,}"
    )

    print(
        "本次需要搜尋約：",
        f"{total_combinations - len(tested_parameters):,}"
    )

    if pool_size is None:
        pool_size = max(
            top_n * 10,
            100
        )

    # 目前只保留前面的候選
    current_best = pd.DataFrame()

    batch = []

    processed = 0

    # --------------------------------------------------------
    # 逐一搜尋
    # --------------------------------------------------------

    for combination in product(
        elevations,
        horizontals,
        spins,
        powers
    ):

        if combination in tested_parameters:
            continue

        batch.append(
            combination
        )

        if len(batch) < chunk_size:
            continue

        # -----------------------------------------------
        # Batch → DataFrame
        # -----------------------------------------------

        batch_df = pd.DataFrame(
            batch,
            columns=feature_columns
        )

        # -----------------------------------------------
        # 模型預測
        # -----------------------------------------------

        probabilities = (
            best_model
            .predict_proba(batch_df)
        )

        # -----------------------------------------------
        # 目標落點機率
        # 如果目標是一個範圍，
        # 就把範圍內所有格子的機率加總
        # -----------------------------------------------

        target_probability = np.zeros(
            len(batch_df)
        )

        for target_label in target_labels:

            target_index = (
                class_to_index[
                    target_label
                ]
            )

            target_probability += (
                probabilities[
                    :,
                    target_index
                ]
            )

        # -----------------------------------------------
        # 模型真正的第一名預測
        # -----------------------------------------------

        best_index = np.argmax(
            probabilities,
            axis=1
        )

        predicted_labels = (
            best_model.classes_[
                best_index
            ]
        )

        # -----------------------------------------------
        # 命中目標
        # -----------------------------------------------

        hit_target = np.isin(
            predicted_labels,
            target_labels
        ).astype(int)

        # -----------------------------------------------
        # 無效機率
        # -----------------------------------------------

        invalid_probability = (
            calculate_invalid_probability(
                probabilities
            )
        )

        # -----------------------------------------------
        # 距離目標
        # -----------------------------------------------

        distances = np.array([
            distance_to_targets(
                pred,
                target_labels
            )
            for pred
            in predicted_labels
        ])

        # -----------------------------------------------
        # 球路條件分數
        # -----------------------------------------------

        condition_score = (
            calculate_condition_score(
                batch_df,
                conditions
            )
            .to_numpy()
        )

        # -----------------------------------------------
        # 組合結果
        # -----------------------------------------------

        batch_result = batch_df.copy()

        batch_result[
            "預測落點"
        ] = predicted_labels

        batch_result[
            "命中目標"
        ] = hit_target

        batch_result[
            "目標落點機率"
        ] = target_probability

        batch_result[
            "無效總機率"
        ] = invalid_probability

        batch_result[
            "目標距離_曼哈頓"
        ] = distances

        batch_result[
            "球路條件分數"
        ] = condition_score

        # -----------------------------------------------
        # 先縮小 Batch
        # -----------------------------------------------

        batch_result = rank_candidate_dataframe(
            batch_result
        ).head(
            pool_size
        )

        # -----------------------------------------------
        # 與上一批合併
        # -----------------------------------------------

        current_best = pd.concat(
            [
                current_best,
                batch_result
            ],
            ignore_index=True
        )

        current_best = (
            rank_candidate_dataframe(
                current_best
            )
            .head(pool_size)
            .copy()
        )

        processed += len(batch)

        print(
            f"\r目前已搜尋：約 "
            f"{processed:,} 組候選",
            end=""
        )

        batch = []

    # --------------------------------------------------------
    # 最後一批
    # --------------------------------------------------------

    if len(batch) > 0:

        batch_df = pd.DataFrame(
            batch,
            columns=feature_columns
        )

        probabilities = (
            best_model
            .predict_proba(batch_df)
        )

        target_probability = np.zeros(
            len(batch_df)
        )

        for target_label in target_labels:

            target_index = (
                class_to_index[
                    target_label
                ]
            )

            target_probability += (
                probabilities[
                    :,
                    target_index
                ]
            )

        best_index = np.argmax(
            probabilities,
            axis=1
        )

        predicted_labels = (
            best_model.classes_[
                best_index
            ]
        )

        hit_target = np.isin(
            predicted_labels,
            target_labels
        ).astype(int)

        invalid_probability = (
            calculate_invalid_probability(
                probabilities
            )
        )

        distances = np.array([
            distance_to_targets(
                pred,
                target_labels
            )
            for pred
            in predicted_labels
        ])

        condition_score = (
            calculate_condition_score(
                batch_df,
                conditions
            )
            .to_numpy()
        )

        batch_result = batch_df.copy()

        batch_result[
            "預測落點"
        ] = predicted_labels

        batch_result[
            "命中目標"
        ] = hit_target

        batch_result[
            "目標落點機率"
        ] = target_probability

        batch_result[
            "無效總機率"
        ] = invalid_probability

        batch_result[
            "目標距離_曼哈頓"
        ] = distances

        batch_result[
            "球路條件分數"
        ] = condition_score

        current_best = pd.concat(
            [
                current_best,
                batch_result
            ],
            ignore_index=True
        )

    # --------------------------------------------------------
    # 最後排序
    # --------------------------------------------------------

    result_df = (
        rank_candidate_dataframe(
            current_best
        )
        .head(top_n)
        .copy()
    )

    # --------------------------------------------------------
    # 額外資訊
    # --------------------------------------------------------

    result_df.insert(
        0,
        "目標落點",
        " / ".join(target_labels)
    )

    result_df.insert(
        1,
        "球路條件",
        " + ".join(conditions)
    )

    print("\n\n✅ 參數搜尋完成")

    print(
        "\n前",
        top_n,
        "組推薦參數："
    )

    display(
        result_df
    )

    return result_df


# ============================================================
# 第 26 格：測試
#
# 這次不是單純找「最高機率」，
# 而是：
#
# ① 先看預測結果是否命中目標
# ② 再看目標機率
# ③ 再看無效機率
# ④ 再看球路條件
# ============================================================

best_parameters = search_best_parameters(
    target_landing="(4,3)",
    ball_condition="標準",
    top_n=20,
    chunk_size=50000
)

display(
    best_parameters
)


# ============================================================
# 第 27 格：
# 多種球路方案
#
# 這裡可以做到：
#
# 同一個落點
# ↓
# 高速球
# 高旋球角度
# 低速球
# 等不同方案
#
# 注意：
# 每呼叫一次都會完整搜尋一次參數空間。
# 若需要一次產生多種方案，
# 下方第 28 格提供更有效率的方式。
# ============================================================

# 例如：
#
# 高速球：
#
# high_speed_result = search_best_parameters(
#     target_landing="(4,3)",
#     ball_condition="高速球",
#     top_n=10
# )
#
#
# 高旋球角度：
#
# high_spin_result = search_best_parameters(
#     target_landing="(4,3)",
#     ball_condition="高旋轉",
#     top_n=10
# )
#
#
# 低速球：
#
# low_speed_result = search_best_parameters(
#     target_landing="(4,3)",
#     ball_condition="低速球",
#     top_n=10
# )


# ============================================================
# 第 28 格：
# 一次搜尋多種「同落點、不同球路」
#
# 這個版本只走一次完整的參數搜尋空間，
# 同時保留不同球路條件各自的最佳候選。
# ============================================================

def search_multiple_route_conditions(
    target_landing,
    ball_conditions,
    top_n=5,
    chunk_size=50000,
    pool_size=100
):

    target_labels = (
        normalize_target_labels(
            target_landing
        )
    )

    conditions = (
        normalize_ball_conditions(
            ball_conditions
        )
    )

    print("==========================================")
    print("多球路方案搜尋")
    print("==========================================")

    print(
        "目標：",
        target_labels
    )

    print(
        "球路方案：",
        conditions
    )

    # --------------------------------------------------------
    # 確認目標存在
    # --------------------------------------------------------

    missing_targets = [
        label
        for label in target_labels
        if label not in class_to_index
    ]

    if missing_targets:

        raise ValueError(
            "以下目標類別不在模型中：\n"
            +
            "\n".join(missing_targets)
        )

    # 每一種球路都保留自己的候選
    route_pools = {
        condition: pd.DataFrame()
        for condition in conditions
    }

    batch = []

    total_combinations = (
        28
        *
        9
        *
        59
        *
        99
    )

    processed = 0

    for combination in product(
        range(1, 29),
        range(1, 10),
        range(1, 60),
        range(1, 100)
    ):

        if combination in tested_parameters:
            continue

        batch.append(
            combination
        )

        if len(batch) < chunk_size:
            continue

        # ----------------------------------------------------
        # 建立 Batch
        # ----------------------------------------------------

        batch_df = pd.DataFrame(
            batch,
            columns=feature_columns
        )

        probabilities = (
            best_model
            .predict_proba(batch_df)
        )

        # ----------------------------------------------------
        # 目標機率
        # ----------------------------------------------------

        target_probability = np.zeros(
            len(batch_df)
        )

        for target_label in target_labels:

            target_probability += (
                probabilities[
                    :,
                    class_to_index[
                        target_label
                    ]
                ]
            )

        # ----------------------------------------------------
        # 預測結果
        # ----------------------------------------------------

        best_index = np.argmax(
            probabilities,
            axis=1
        )

        predicted_labels = (
            best_model.classes_[
                best_index
            ]
        )

        hit_target = np.isin(
            predicted_labels,
            target_labels
        ).astype(int)

        # ----------------------------------------------------
        # 無效機率
        # ----------------------------------------------------

        invalid_probability = (
            calculate_invalid_probability(
                probabilities
            )
        )

        # ----------------------------------------------------
        # 目標距離
        # ----------------------------------------------------

        distances = np.array([
            distance_to_targets(
                pred,
                target_labels
            )
            for pred in predicted_labels
        ])

        # ----------------------------------------------------
        # 基礎結果
        # ----------------------------------------------------

        base_result = batch_df.copy()

        base_result[
            "預測落點"
        ] = predicted_labels

        base_result[
            "命中目標"
        ] = hit_target

        base_result[
            "目標落點機率"
        ] = target_probability

        base_result[
            "無效總機率"
        ] = invalid_probability

        base_result[
            "目標距離_曼哈頓"
        ] = distances

        # ----------------------------------------------------
        # 每一種球路分別排序
        # ----------------------------------------------------

        for condition in conditions:

            condition_score = (
                calculate_condition_score(
                    batch_df,
                    [condition]
                )
                .to_numpy()
            )

            route_result = (
                base_result.copy()
            )

            route_result[
                "球路條件分數"
            ] = condition_score

            route_result = (
                rank_candidate_dataframe(
                    route_result
                )
                .head(pool_size)
            )

            route_pools[condition] = (
                pd.concat(
                    [
                        route_pools[condition],
                        route_result
                    ],
                    ignore_index=True
                )
            )

            route_pools[condition] = (
                rank_candidate_dataframe(
                    route_pools[condition]
                )
                .head(pool_size)
                .copy()
            )

        processed += len(batch)

        print(
            f"\r目前已搜尋約 "
            f"{processed:,} 組候選",
            end=""
        )

        batch = []

    # --------------------------------------------------------
    # 最後一批
    # --------------------------------------------------------

    if len(batch) > 0:

        batch_df = pd.DataFrame(
            batch,
            columns=feature_columns
        )

        probabilities = (
            best_model
            .predict_proba(batch_df)
        )

        target_probability = np.zeros(
            len(batch_df)
        )

        for target_label in target_labels:

            target_probability += (
                probabilities[
                    :,
                    class_to_index[
                        target_label
                    ]
                ]
            )

        best_index = np.argmax(
            probabilities,
            axis=1
        )

        predicted_labels = (
            best_model.classes_[
                best_index
            ]
        )

        hit_target = np.isin(
            predicted_labels,
            target_labels
        ).astype(int)

        invalid_probability = (
            calculate_invalid_probability(
                probabilities
            )
        )

        distances = np.array([
            distance_to_targets(
                pred,
                target_labels
            )
            for pred in predicted_labels
        ])

        base_result = batch_df.copy()

        base_result[
            "預測落點"
        ] = predicted_labels

        base_result[
            "命中目標"
        ] = hit_target

        base_result[
            "目標落點機率"
        ] = target_probability

        base_result[
            "無效總機率"
        ] = invalid_probability

        base_result[
            "目標距離_曼哈頓"
        ] = distances

        for condition in conditions:

            condition_score = (
                calculate_condition_score(
                    batch_df,
                    [condition]
                )
                .to_numpy()
            )

            route_result = (
                base_result.copy()
            )

            route_result[
                "球路條件分數"
            ] = condition_score

            route_result = (
                rank_candidate_dataframe(
                    route_result
                )
                .head(pool_size)
            )

            route_pools[condition] = (
                pd.concat(
                    [
                        route_pools[condition],
                        route_result
                    ],
                    ignore_index=True
                )
            )

            route_pools[condition] = (
                rank_candidate_dataframe(
                    route_pools[condition]
                )
                .head(pool_size)
                .copy()
            )

    # --------------------------------------------------------
    # 最後整理
    # --------------------------------------------------------

    final_results = {}

    for condition in conditions:

        result = (
            rank_candidate_dataframe(
                route_pools[condition]
            )
            .head(top_n)
            .copy()
        )

        result.insert(
            0,
            "目標落點",
            " / ".join(target_labels)
        )

        result.insert(
            1,
            "球路條件",
            condition
        )

        final_results[condition] = (
            result
        )

        print(
            f"\n\n========== {condition} =========="
        )

        display(
            result
        )

    print(
        "\n✅ 多球路搜尋完成"
    )

    return final_results


# ============================================================
# 第 29 格：
# 一次產生同一落點的多種球路
#
# 注意：
# 這裡只要執行一次完整搜尋，
# 就會同時得到：
#
# ① 高速球
# ② 高旋轉
# ③ 低速球
#
# ============================================================

multi_route_results = (
    search_multiple_route_conditions(
        target_landing="(4,3)",

        ball_conditions=[
            "高速球",
            "高旋轉",
            "低速球"
        ],

        top_n=5,

        chunk_size=50000,

        pool_size=100
    )
)


# ============================================================
# 第 30 格：
# 指定「目標範圍」
#
# 例如你想要球落在：
#
# (3,2)
# (3,3)
# (4,2)
# (4,3)
#
# 都算成功。
#
# 這就符合你原本寫的：
# 「精準落在指定位置或指定範圍」
# ============================================================

# 範例：
#
# target_area_results = search_best_parameters(
#
#     target_landing=[
#         "(3,2)",
#         "(3,3)",
#         "(4,2)",
#         "(4,3)"
#     ],
#
#     ball_condition="高速球",
#
#     top_n=20
# )
#
# display(
#     target_area_results
# )


# ============================================================
# 第 31 格：
# 建立「多組不同參數」的去重 / 多樣化選擇
#
# 防止前 5 組全部幾乎一樣。
# ============================================================

def select_diverse_solutions(
    result_df,
    n=5,
    min_normalized_distance=0.08
):

    if result_df.empty:
        return result_df.copy()

    working = result_df.copy()

    selected_rows = []

    # --------------------------------------------------------
    # 四個參數轉成 0~1
    # --------------------------------------------------------

    def get_normalized_vector(row):

        vector = []

        for col, (
            min_value,
            max_value
        ) in parameter_limits.items():

            value = (
                row[col]
            )

            normalized = (
                value - min_value
            ) / (
                max_value - min_value
            )

            vector.append(
                normalized
            )

        return np.array(
            vector,
            dtype=float
        )

    # --------------------------------------------------------
    # 由排序好的結果逐一挑選
    # --------------------------------------------------------

    for _, row in working.iterrows():

        candidate_vector = (
            get_normalized_vector(
                row
            )
        )

        is_diverse = True

        for selected_row in selected_rows:

            selected_vector = (
                get_normalized_vector(
                    selected_row
                )
            )

            distance = np.linalg.norm(
                candidate_vector
                -
                selected_vector
            )

            if (
                distance
                <
                min_normalized_distance
            ):

                is_diverse = False

                break

        if is_diverse:

            selected_rows.append(
                row
            )

        if len(selected_rows) >= n:
            break

    if len(selected_rows) == 0:
        return working.head(n).copy()

    return pd.DataFrame(
        selected_rows
    ).reset_index(
        drop=True
    )


# ============================================================
# 第 32 格：
# 取出「同落點、不同球路」且參數不太相同的方案
# ============================================================

diverse_route_results = {}

for condition, result in (
    multi_route_results.items()
):

    diverse_result = (
        select_diverse_solutions(
            result,
            n=5,
            min_normalized_distance=0.08
        )
    )

    diverse_route_results[
        condition
    ] = diverse_result

    print(
        f"\n========== {condition}：多樣化方案 =========="
    )

    display(
        diverse_result
    )


# ============================================================
# 第 33 格：
# 顯示「使用者真正可以拿來操作」的推薦格式
# ============================================================

def print_recommended_solutions(
    result_dict
):

    print("\n")
    print("================================================")
    print("          桌球發球機智慧控制推薦")
    print("================================================")

    for condition, result in (
        result_dict.items()
    ):

        print(
            f"\n【球路方案：{condition}】"
        )

        if result.empty:

            print(
                "沒有找到符合條件的方案。"
            )

            continue

        for i, (_, row) in enumerate(
            result.iterrows(),
            start=1
        ):

            print(
                f"\n方案 {i}"
            )

            print(
                "仰角：",
                int(row["仰角角度"])
            )

            print(
                "左右位置：",
                int(row["左右位置"])
            )

            print(
                "旋球角度：",
                int(row["旋球角度"])
            )

            print(
                "出球強度：",
                int(row["出球強度"])
            )

            print(
                "模型預測落點：",
                row["預測落點"]
            )

            print(
                "命中目標：",
                "✅ 是"
                if row["命中目標"] == 1
                else "❌ 否"
            )

            print(
                "目標落點機率：",
                f"{row['目標落點機率'] * 100:.2f}%"
            )

            print(
                "無效總機率：",
                f"{row['無效總機率'] * 100:.2f}%"
            )

            print(
                "目標距離：",
                row["目標距離_曼哈頓"]
            )

            print(
                "球路條件分數：",
                f"{row['球路條件分數'] * 100:.2f}%"
            )

    print("\n================================================")


print_recommended_solutions(
    diverse_route_results
)


# ============================================================
# 第 34 格：
# 輸出模型測試結果
# ============================================================

evaluation_df = pd.DataFrame({

    "測試編號":
        groups_test.values,

    "仰角角度":
        X_test["仰角角度"].values,

    "左右位置":
        X_test["左右位置"].values,

    "旋球角度":
        X_test["旋球角度"].values,

    "出球強度":
        X_test["出球強度"].values,

    "實際落點":
        y_test.values,

    "模型預測":
        best_pred
})

evaluation_filename = (
    "model_test_results.csv"
)

evaluation_df.to_csv(
    evaluation_filename,
    index=False,
    encoding="utf-8-sig"
)

print(
    "✅ 測試結果已儲存：",
    evaluation_filename
)

files.download(
    evaluation_filename
)


# ============================================================
# 第 35 格：
# 儲存模型
# ============================================================

model_package = {

    "model":
        best_model,

    "model_name":
        best_model_name,

    "feature_columns":
        feature_columns,

    "class_names":
        list(
            best_model.classes_
        ),

    "parameter_limits":
        parameter_limits,

    "valid_labels":
        VALID_LABELS,

    "invalid_labels":
        INVALID_LABELS,

    "all_labels":
        ALL_LABELS,

    "num_classes":
        len(
            best_model.classes_
        ),

    "ball_condition_map":
        BALL_CONDITION_MAP,

    "tested_parameter_count":
        len(
            tested_parameters
        ),

    "training_info": {

        "training_samples":
            int(
                len(X_train)
            ),

        "testing_samples":
            int(
                len(X_test)
            ),

        "training_groups":
            int(
                groups_train.nunique()
            ),

        "testing_groups":
            int(
                groups_test.nunique()
            ),

        "accuracy":
            float(
                best_row["Accuracy"]
            ),

        "balanced_accuracy":
            float(
                best_row[
                    "Balanced Accuracy"
                ]
            ),

        "macro_f1":
            float(
                best_row["Macro F1"]
            ),

        "weighted_f1":
            float(
                best_row["Weighted F1"]
            )
    }
}


model_filename = (
    "table_tennis_ball_landing_model.pkl"
)

joblib.dump(
    model_package,
    model_filename
)

print(
    "✅ 模型已儲存：",
    model_filename
)


# ============================================================
# 第 36 格：下載模型
# ============================================================

files.download(
    model_filename
)


# ============================================================
# 第 37 格：
# 儲存 Feature Importance
# ============================================================

importance_filename = (
    "feature_importance.csv"
)

importance_df.to_csv(
    importance_filename,
    index=False,
    encoding="utf-8-sig"
)

print(
    "✅ Feature Importance 已儲存：",
    importance_filename
)

files.download(
    importance_filename
)


# ============================================================
# 第 38 格：
# 儲存本次推薦結果
# ============================================================

all_recommendations = []

for condition, result in (
    diverse_route_results.items()
):

    temp = result.copy()

    temp[
        "球路方案"
    ] = condition

    all_recommendations.append(
        temp
    )

if len(all_recommendations) > 0:

    recommendation_df = pd.concat(
        all_recommendations,
        ignore_index=True
    )

else:

    recommendation_df = pd.DataFrame()


recommendation_filename = (
    "recommended_parameter_solutions.csv"
)

recommendation_df.to_csv(
    recommendation_filename,
    index=False,
    encoding="utf-8-sig"
)

print(
    "✅ 推薦參數已儲存：",
    recommendation_filename
)

files.download(
    recommendation_filename
)


# ============================================================
# 第 39 格：
# 完成
# ============================================================

print("\n")
print("==========================================================")
print("✅ 整個桌球自動發球機模型流程完成")
print("==========================================================")

print("\n目前已完成：")

print("① 四個參數 → 落點預測")
print("② Random Forest / Extra Trees 比較")
print("③ Group Train/Test Split")
print("④ 29 類結果分類")
print("⑤ 落點空間距離評估")
print("⑥ 指定目標落點 → 反推四參數")
print("⑦ 指定目標範圍 → 反推四參數")
print("⑧ 四個參數全部一起搜尋")
print("⑨ 第一優先：目標落點精準度")
print("⑩ 第二優先：球路條件")
print("⑪ 高速 / 高強度方案")
print("⑫ 高旋球角度方案")
print("⑬ 低速 / 低強度方案")
print("⑭ 多組同落點、不同球路方案")
print("⑮ 排除第一階段已測參數")
print("⑯ 輸出模型與推薦參數")

print("\n⚠️ 最後仍需要：")
print("實際使用發球機驗證推薦參數，")
print("再把實測結果加入資料集，進行下一輪模型更新。")

print("\n✅ 模型訓練與智慧搜尋架構建立完成。")