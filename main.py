from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np

app = FastAPI()

# モデル読み込み
MODEL_PATH = "ws0t4/trained_emotions_classification_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# クライアントからの入力形式
class InputText(BaseModel):
    text: str

# ラベル名の順番（この順番で回転させる）
LABEL_ORDER = [
    "LABEL_0", "LABEL_1", "LABEL_2", "LABEL_3",
    "LABEL_4", "LABEL_5", "LABEL_6", "LABEL_7"
]

# 回転行列（対応する8方向）
ROTATE_MATRICES = [
    np.array([[0, -1], [1, 0]]),  # 90度
    np.array([[0, 1], [-1, 0]]),  # 270度
    np.array([[np.cos(np.deg2rad(135)), -np.sin(np.deg2rad(135))],
              [np.sin(np.deg2rad(135)),  np.cos(np.deg2rad(135))]]),
    np.array([[np.cos(np.deg2rad(315)), -np.sin(np.deg2rad(315))],
              [np.sin(np.deg2rad(315)),  np.cos(np.deg2rad(315))]]),
    np.array([[-1, 0], [0, -1]]),  # 180度
    np.array([[1, 0], [0, 1]]),    # 0度
    np.array([[np.cos(np.deg2rad(225)), -np.sin(np.deg2rad(225))],
              [np.sin(np.deg2rad(225)),  np.cos(np.deg2rad(225))]]),
    np.array([[np.cos(np.deg2rad(45)), -np.sin(np.deg2rad(45))],
              [np.sin(np.deg2rad(45)),  np.cos(np.deg2rad(45))]])
]

@app.post("/predict")
def predict(input: InputText):
    # 推論結果の取得
    result = pipe(input.text)
    print(result)
    raw_result = result[0]

    # ラベルとスコアを辞書に
    result_dict = {item["label"]: item["score"] for item in raw_result}

    # ラベルの順序を固定して、スコアをリストに
    scores = [result_dict.get(label, 0.0) for label in LABEL_ORDER]

    emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']

    # 2D座標に変換
    coordinates = []
    for prob, rotater in zip(scores, ROTATE_MATRICES):
        vec = np.array([prob, 0])
        rotated = rotater @ vec
        coordinates.append(rotated.tolist())

    coordinates_np = np.array(coordinates)

    # 合成座標の計算
    x_comp = float(np.sum(coordinates_np[:, 0]))
    y_comp = float(np.sum(coordinates_np[:, 1]))

    for label,value in zip(emotion_names_jp,result_dict.values()):
        print(label + ":" + str(value))

    return {
        "coordinates": coordinates,
        "combined_vector": {"x": x_comp, "y": y_comp}
    }

