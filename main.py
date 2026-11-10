"""Matchmaking: predict a user's sex from OkCupid profile answers.

Data: profiles.csv from the OkCupid profiles dataset (place it next to this file).
"""
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "profiles.csv"

# ---------------------------------------------------------------------------
# Mappings from free-text answers to codes
# ---------------------------------------------------------------------------
BODY = {"thin": 0, "used up": 0, "a little extra": 1, "full figured": 1, "overweight": 1,
        "average": 2, "fit": 3, "athletic": 3, "jacked": 4, "skinny": 0, "curvy": 1}
DRINKS = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
SMOKES = {"no": 0, "sometimes": 1, "when drinking": 2, "trying to quit": 3, "yes": 4}
DRUGS = {"never": 0, "sometimes": 1, "often": 2}
# Highest finished degree: high school 0, two-year college 1, college 2, masters 3,
# ph.d / med / law 4. "working on" and "dropped out" count as the level below.
EDUCATION_LEVELS = {"high school": 0, "two-year college": 1, "college/university": 2,
                    "masters program": 3, "ph.d program": 4, "med school": 4, "law school": 4}
DIET_GROUPS = {"halal": "religious", "kosher": "religious", "vegan": "vegan",
               "vegetarian": "vegetarian", "anything": "anything", "other": "other"}


def education_code(text):
    if not isinstance(text, str) or "space camp" in text:
        return np.nan
    for name, level in EDUCATION_LEVELS.items():
        if name in text:
            finished = text.startswith("graduated from") or text == name
            return level if finished else max(level - 1, 0)
    return np.nan


def diet_group(text):
    if not isinstance(text, str):
        return np.nan
    return DIET_GROUPS.get(text.split()[-1], np.nan)


# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

df["sign_simple"] = df["sign"].str.split().str.get(0)
df["religion_simple"] = df["religion"].str.split().str.get(0)
df["diet_simple"] = df["diet"].map(diet_group)
df["body_code"] = df["body_type"].map(BODY)
df["drinks_code"] = df["drinks"].map(DRINKS)
df["smokes_code"] = df["smokes"].map(SMOKES)
df["drugs_code"] = df["drugs"].map(DRUGS)
df["education_code"] = df["education"].map(education_code)

# Text features from the ten profile essays
essays = df[[f"essay{i}" for i in range(10)]].fillna("").agg(" ".join, axis=1)
df["essay_len"] = essays.str.len()
df["avg_word_length"] = (essays.str.len() + 1) / (essays.str.split().str.len() + 1)

NUMERIC = ["body_code", "drinks_code", "smokes_code", "drugs_code", "education_code",
           "essay_len", "avg_word_length"]
CATEGORICAL = ["sign_simple", "religion_simple", "diet_simple"]

df = df.dropna(subset=NUMERIC + CATEGORICAL + ["sex"])
X = df[NUMERIC + CATEGORICAL]
y = df["sex"]

print(f"Profiles used: {len(df):,}")
print("Class balance:", y.value_counts(normalize=True).round(3).to_dict(), "\n")

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Numbers are scaled, categories one-hot encoded. Fitted on the training set only.
preprocess = ColumnTransformer([
    ("num", MinMaxScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])

# Hyperparameters from the tuning runs (depth sweep, k sweep, SVM grid search)
models = {
    "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=34),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=34),
    "KNN": KNeighborsClassifier(n_neighbors=20),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(kernel="rbf", C=100, gamma=0.01),
}

# ---------------------------------------------------------------------------
# Train and evaluate on the held-out test set
# ---------------------------------------------------------------------------
results = []
for name, model in models.items():
    pipeline = make_pipeline(preprocess, model).fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    results.append({
        "model": name,
        "train_acc": pipeline.score(x_train, y_train),
        "test_acc": pipeline.score(x_test, y_test),
        "balanced_acc": balanced_accuracy_score(y_test, predictions),
    })
    print(f"=== {name} ===")
    print(classification_report(y_test, predictions, zero_division=0))

summary = pd.DataFrame(results).set_index("model")
print((summary * 100).round(1).to_string())
