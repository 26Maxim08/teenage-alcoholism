"""
Веб-интерфейс для модели определения риска злоупотребления алкоголем.
Запуск:  python app.py  ->  http://127.0.0.1:5000
"""
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

bundle = joblib.load("model.joblib")
pipeline = bundle["pipeline"]
schema = bundle["schema"]
columns = bundle["columns"]
numeric = set(bundle["numeric"])

# группировка полей формы по смыслу (только для удобства интерфейса)
GROUPS = {
    "О студенте": ["sex", "age", "address", "famsize"],
    "Семья": ["Medu", "Fedu", "famrel", "nursery"],
    "Учёба": ["studytime", "failures", "activities", "absences", "higher", "internet"],
    "Образ жизни": ["goout", "freetime", "romantic"],
}
schema_by_name = {f["name"]: f for f in schema}
grouped = [(g, [schema_by_name[n] for n in names if n in schema_by_name])
           for g, names in GROUPS.items()]

app = Flask(__name__)


@app.route("/")
def index():
    return render_template(
        "index.html",
        grouped=grouped,
        roc_auc=bundle["roc_auc"],
        app_threshold=bundle["app_threshold"],
        recall_threshold=bundle["recall_threshold"],
        model_name=bundle["selected_model"],
        metrics=bundle["metrics_default"],
    )


def row_from_form(data):
    """Собирает однострочный DataFrame в правильном порядке колонок."""
    row = {}
    for col in columns:
        val = data.get(col, schema_by_name[col]["default"])
        if col in numeric:
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = float(schema_by_name[col]["default"])
        else:
            val = str(val)
        row[col] = val
    return pd.DataFrame([row], columns=columns)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    X = row_from_form(data)
    proba = float(pipeline.predict_proba(X)[0, 1])
    return jsonify({
        "probability": round(proba, 4),
        "app_threshold": bundle["app_threshold"],
        "recall_threshold": bundle["recall_threshold"],
        "class_names": bundle["class_names"],
    })


@app.route("/sample")
def sample():
    """Случайный реальный студент из датасета — для быстрого демо."""
    df = pd.read_csv("data/student-mat.csv").drop(columns=["Dalc", "Walc"])
    rec = df.sample(1).iloc[0].to_dict()
    return jsonify({k: (int(v) if k in numeric else str(v)) for k, v in rec.items()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
