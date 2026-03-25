import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

csv_path = os.path.join("data", "online_shoppers_intention.csv")
df = pd.read_csv(csv_path)

print("Путь к CSV:", csv_path)
print("Размер датасета:", df.shape)
print(df.head())

target_col = "Revenue"
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

print("Доля положительного класса (Revenue=1):", y.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(
    "Размер обучающей выборки:", X_train.shape, "Размер тестовой выборки:", X_test.shape
)
print(
    "Доля класса 1 в обучающей выборке:",
    y_train.mean(),
    "в тестовой выборке:",
    y_test.mean(),
)

cat_cols = X.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

print("Количество числовых признаков:", len(num_cols))
print("Количество категориальных признаков:", len(cat_cols))

# 5) Модели
models = {
    "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    ),
    "HistGB": HistGradientBoostingClassifier(random_state=42),
}


def evaluate(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"Точность:                  {acc:.4f}")
    print(f"Точность положительного класса:       {prec:.4f}")
    print(f"Полнота:                     {rec:.4f}")
    print(f"F1-мера:                              {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("Матрица ошибок:\n", cm)
    print(
        "\nОтчет классификации:\n",
        classification_report(y_test, y_pred, zero_division=0),
    )


trained = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    trained[name] = pipe
    evaluate(name, pipe, X_test, y_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"{name} средний CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

name = "LogReg"
pipe = trained[name]


probs = pipe.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0.05, 0.95, 19)

best_t, best_p, best_r, best_f1 = None, 0, 0, 0
for t in thresholds:
    pred = (probs >= t).astype(int)
    p = precision_score(y_test, pred, zero_division=0)
    r = recall_score(y_test, pred, zero_division=0)
    f = f1_score(y_test, pred, zero_division=0)
    if f > best_f1:
        best_t, best_p, best_r, best_f1 = t, p, r, f

print("\nЛучший порог вероятности по F1 :", best_t)
print(f"Точность={best_p:.4f}, Полнота ={best_r:.4f}, F1={best_f1:.4f}")
