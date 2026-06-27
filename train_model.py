import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, RepeatedStratifiedKFold, StratifiedKFold,
    cross_val_score, cross_val_predict)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve)

RANDOM_STATE = 42
DATA_PATH = "data/student-mat.csv"
THRESHOLD_RULE = 14         # порог weekly-индекса для класса "риск"
TIE_MARGIN = 0.01           # модели в пределах этого зазора по AUC считаем равными

LABELS = {
    "school": "Школа", "sex": "Пол", "age": "Возраст",
    "address": "Тип местности", "famsize": "Размер семьи",
    "Pstatus": "Родители живут", "Medu": "Образование матери (0–4)",
    "Fedu": "Образование отца (0–4)", "Mjob": "Работа матери",
    "Fjob": "Работа отца", "reason": "Причина выбора школы",
    "guardian": "Опекун", "traveltime": "Время в пути до школы (1–4)",
    "studytime": "Время на учёбу в неделю (1–4)",
    "failures": "Проваленные предметы (0–3)",
    "schoolsup": "Доп. поддержка от школы", "famsup": "Поддержка семьи в учёбе",
    "paid": "Платные доп. занятия", "activities": "Внеклассные активности",
    "nursery": "Посещал детсад", "higher": "Хочет высшее образование",
    "internet": "Интернет дома", "romantic": "В отношениях",
    "famrel": "Отношения в семье (1–5)", "freetime": "Свободное время (1–5)",
    "goout": "Прогулки с друзьями (1–5)", "health": "Состояние здоровья (1–5)",
    "absences": "Пропуски занятий", "G1": "Оценка, 1 период (0–20)",
    "G2": "Оценка, 2 период (0–20)", "G3": "Итоговая оценка (0–20)",
}
OPTION_LABELS = {
    "school": {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"},
    "sex": {"F": "Женский", "M": "Мужской"},
    "address": {"U": "Город", "R": "Село"},
    "famsize": {"GT3": "Больше 3 человек", "LE3": "3 и меньше"},
    "Pstatus": {"T": "Вместе", "A": "Раздельно"},
    "Mjob": {"at_home": "Дома", "health": "Здравоохранение", "services": "Сфера услуг",
             "teacher": "Учитель", "other": "Другое"},
    "Fjob": {"at_home": "Дома", "health": "Здравоохранение", "services": "Сфера услуг",
             "teacher": "Учитель", "other": "Другое"},
    "reason": {"course": "Программа курса", "home": "Близко к дому",
               "reputation": "Репутация школы", "other": "Другое"},
    "guardian": {"mother": "Мать", "father": "Отец", "other": "Другое"},
}
for _c in ["schoolsup", "famsup", "paid", "activities", "nursery", "higher",
           "internet", "romantic"]:
    OPTION_LABELS[_c] = {"yes": "Да", "no": "Нет"}

# --- набор признаков (отобран по permutation importance + интуитивные поля) ---
FEATURES = ["goout", "failures", "famrel", "address", "famsize", "sex",
            "activities", "studytime", "age", "freetime", "romantic", "absences",
            "nursery", "internet", "higher", "Medu", "Fedu"]

# --- расшифровка порядковых шкал в человеческие подписи ---
EDU = {0: "Нет", 1: "Начальное (4 кл.)", 2: "5–9 классы",
       3: "Среднее", 4: "Высшее"}
VALUE_MEANINGS = {
    "goout": {1: "Очень редко", 2: "Редко", 3: "Иногда", 4: "Часто", 5: "Очень часто"},
    "famrel": {1: "Очень плохие", 2: "Плохие", 3: "Средние", 4: "Хорошие", 5: "Отличные"},
    "studytime": {1: "Менее 2 ч/нед", 2: "2–5 ч/нед", 3: "5–10 ч/нед", 4: "Более 10 ч/нед"},
    "failures": {0: "Нет", 1: "1 предмет", 2: "2 предмета", 3: "3 и более"},
    "freetime": {1: "Очень мало", 2: "Мало", 3: "Средне", 4: "Много", 5: "Очень много"},
    "Medu": EDU, "Fedu": EDU,
}
# более понятные подписи полей
LABELS.update({
    "goout": "Как часто гуляет с друзьями",
    "failures": "Проваленные предметы в прошлом",
    "famrel": "Отношения в семье",
    "studytime": "Время на учёбу в неделю",
    "activities": "Ходит на внеклассные занятия",
    "freetime": "Свободное время после школы",
    "romantic": "В романтических отношениях",
    "absences": "Пропуски занятий (дней)",
    "age": "Возраст",
    "nursery": "Посещал детский сад",
    "internet": "Интернет дома",
    "higher": "Хочет получить высшее",
    "Medu": "Образование матери",
    "Fedu": "Образование отца",
})


def prettify(raw_name):
    """num__goout -> 'Прогулки...'; cat__sex_M -> 'Пол = Мужской'."""
    body = raw_name.split("__", 1)[1]
    if raw_name.startswith("num__"):
        return LABELS.get(body, body)
    # категориальный: base_value
    for col in OPTION_LABELS:
        if body.startswith(col + "_"):
            val = body[len(col) + 1:]
            vlab = OPTION_LABELS[col].get(val, val)
            return f"{LABELS.get(col, col)} = {vlab}"
    return body


def main():
    df = pd.read_csv(DATA_PATH)
    weekly = df["Dalc"] * 5 + df["Walc"] * 2
    y = (weekly >= THRESHOLD_RULE).astype(int)
    X = df[FEATURES].copy()

    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [c for c in X.columns if c not in numeric]
    print(f"Объектов: {len(X)} | признаков: {X.shape[1]} "
          f"({len(numeric)} числовых, {len(categorical)} категориальных)")
    print(f"Класс 'риск': {int(y.sum())} ({y.mean()*100:.1f}%) | "
          f"'норма': {int((1-y).sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    candidates = {
        "Логистическая регрессия": LogisticRegression(
            max_iter=2000, class_weight="balanced"),
        "Случайный лес": RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            random_state=RANDOM_STATE),
        "Градиентный бустинг": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE),
    }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    print("\nСравнение моделей ")
    cv_results = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_results[name] = (s.mean(), s.std())
        print(f"  {name:<24} ROC-AUC = {s.mean():.3f} ± {s.std():.3f}")

    # Правило выбора: лучший по AUC; но если логистическая регрессия в пределах
    # TIE_MARGIN от лидера — выбираем её (интерпретируемость при равном качестве).
    top = max(cv_results, key=lambda k: cv_results[k][0])
    lr = "Логистическая регрессия"
    if cv_results[top][0] - cv_results[lr][0] <= TIE_MARGIN:
        best_name = lr
    else:
        best_name = top
    print(f"\nВыбрана модель: {best_name}")

    best = Pipeline([("pre", pre), ("clf", candidates[best_name])])

    # --- подбор рабочего порога по F1 на out-of-fold предсказаниях трейна ---
    oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(best, X_train, y_train, cv=oof_cv, method="predict_proba")[:, 1]
    grid = np.linspace(0.10, 0.80, 71)
    op_threshold = float(max(grid, key=lambda t: f1_score(y_train, (oof >= t).astype(int))))
    print(f"Рабочий порог (макс. F1 по CV): {op_threshold:.2f}")

    best.fit(X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]

    def block(thr):
        pred = (proba >= thr).astype(int)
        return {
            "threshold": round(float(thr), 2),
            "accuracy": round(float(accuracy_score(y_test, pred)), 3),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 3),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 3),
        }

    roc_auc = round(float(roc_auc_score(y_test, proba)), 3)
    m_default = block(0.5)
    m_op = block(op_threshold)
    print(f"\nROC-AUC (тест): {roc_auc}")
    print("Порог 0.5        :", m_default)
    print("Порог 0.3        :", m_op)
    pred_def = (proba >= 0.5).astype(int)
    print("\nОтчёт при пороге 0.5:")
    print(classification_report(y_test, pred_def, target_names=["норма", "риск"]))

    # --- графики ---
    cm = confusion_matrix(y_test, pred_def)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["норма", "риск"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["норма", "риск"])
    ax.set_xlabel("Предсказано"); ax.set_ylabel("Факт")
    ax.set_title("Матрица ошибок (порог 0.50)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.tight_layout(); fig.savefig("reports/confusion_matrix.png", dpi=130); plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(fpr, tpr, lw=2, color="#2b6cb0", label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC-кривая")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig("reports/roc_curve.png", dpi=130); plt.close(fig)

    prec, rec, thr = precision_recall_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(rec, prec, lw=2, color="#805ad5")
    ax.scatter([m_default["recall"]], [m_default["precision"]], color="#2b6cb0", zorder=5,
               label="порог 0.50")
    ax.scatter([m_op["recall"]], [m_op["precision"]], color="#c97b2c", zorder=5,
               label=f"скрининг (порог {op_threshold:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig("reports/precision_recall.png", dpi=130); plt.close(fig)

    # важность признаков (модель-агностично)
    perm = permutation_importance(best, X_test, y_test, n_repeats=20,
                                  random_state=RANDOM_STATE, scoring="roc_auc")
    order = np.argsort(perm.importances_mean)[::-1][:15]
    feats = [LABELS.get(X.columns[i], X.columns[i]) for i in order]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(feats))[::-1], perm.importances_mean[order], color="#c97b2c")
    ax.set_yticks(range(len(feats))[::-1]); ax.set_yticklabels(feats, fontsize=9)
    ax.set_xlabel("Падение ROC-AUC при перемешивании признака")
    ax.set_title("Топ-15 важных признаков (permutation importance)")
    fig.tight_layout(); fig.savefig("reports/feature_importance.png", dpi=130); plt.close(fig)

    # если модель линейная — нарисуем направленные факторы риска
    clf = best.named_steps["clf"]
    if hasattr(clf, "coef_"):
        names = best.named_steps["pre"].get_feature_names_out()
        coef = clf.coef_[0]
        idx = np.argsort(np.abs(coef))[::-1][:12]
        idx = idx[np.argsort(coef[idx])]  # отсортировать по знаку для картинки
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ["#2f855a" if c < 0 else "#c53030" for c in coef[idx]]
        ax.barh(range(len(idx)), coef[idx], color=colors)
        ax.set_yticks(range(len(idx))); ax.set_yticklabels([prettify(names[i]) for i in idx], fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Коэффициент (← снижает риск | повышает риск →)")
        ax.set_title("Факторы риска (коэффициенты логрегрессии)")
        fig.tight_layout(); fig.savefig("reports/risk_factors.png", dpi=130); plt.close(fig)

    # --- схема признаков для веб-формы ---
    schema = []
    for col in X.columns:
        if col in VALUE_MEANINGS:  # порядковая шкала -> выпадающий список с подписями
            meanings = VALUE_MEANINGS[col]
            opts = [{"value": int(v), "label": meanings[v]} for v in sorted(meanings)]
            schema.append({"name": col, "label": LABELS.get(col, col), "type": "select",
                           "options": opts, "default": int(round(X[col].median()))})
        elif col in numeric:
            schema.append({"name": col, "label": LABELS.get(col, col), "type": "number",
                           "min": int(X[col].min()), "max": int(X[col].max()),
                           "default": int(round(X[col].median()))})
        else:
            opts = [{"value": v, "label": OPTION_LABELS.get(col, {}).get(v, v)}
                    for v in sorted(X[col].dropna().unique())]
            schema.append({"name": col, "label": LABELS.get(col, col), "type": "select",
                           "options": opts, "default": X[col].mode().iloc[0]})

    bundle = {
        "pipeline": best,
        "schema": schema,
        "columns": list(X.columns),
        "numeric": numeric,
        "app_threshold": 0.5,
        "recall_threshold": op_threshold,
        "roc_auc": roc_auc,
        "metrics_default": m_default,
        "metrics_operating": m_op,
        "cv_results": {k: [round(m, 3), round(s, 3)] for k, (m, s) in cv_results.items()},
        "selected_model": best_name,
        "threshold_rule": "weekly = Dalc*5 + Walc*2 ; класс 'риск' если weekly >= 14",
        "class_names": {"0": "Низкий риск", "1": "Группа риска"},
    }
    joblib.dump(bundle, "model.joblib")
    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: bundle[k] for k in
                   ["selected_model", "roc_auc", "app_threshold", "recall_threshold",
                    "metrics_default", "metrics_operating", "cv_results"]},
                  f, ensure_ascii=False, indent=2)
    print("\nСохранено: model.joblib, reports/*.png, reports/metrics.json")


if __name__ == "__main__":
    main()
