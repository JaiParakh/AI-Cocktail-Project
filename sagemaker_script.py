import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


FLAVOR_CATEGORIES = ['bittersweet', 'citrus', 'creamy', 'floral', 'fruity', 'herbal', 'savoury', 'spicy', 'sweet']

# Custom Threshold after a lot of experiments.
OPTIMAL_THRESHOLDS = {
    'bittersweet': 0.50,
    'citrus': 0.50,
    'creamy': 0.60,
    'floral': 0.90,
    'fruity': 0.60,
    'herbal': 0.55,
    'savoury': 0.80,
    'spicy': 0.75,
    'sweet': 0.80,
}

SEED = 66

# Custom class for inference because of the optimization mentioned in the report.
class CocktailTastePredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, models_dict, thresholds, feature_columns, flavor_categories):
        self.models_dict = models_dict
        self.thresholds = thresholds
        self.feature_columns = feature_columns
        self.flavor_categories = flavor_categories

    def _to_array(self, X):
        if isinstance(X, np.ndarray):
            return X
        if isinstance(X, list):
            return np.array([[row.get(c, 0.0) for c in self.feature_columns] for row in X])
        if isinstance(X, dict):
            return np.array([[X.get(c, 0.0) for c in self.feature_columns]])
        if isinstance(X, pd.DataFrame):
            missing = set(self.feature_columns) - set(X.columns)
            if missing:
                X = pd.concat([X, pd.DataFrame(0, index=X.index, columns=list(missing))], axis=1)
            return X[self.feature_columns].fillna(0.0).values
        raise ValueError("Input must be ndarray, list, dict, or DataFrame")

    def predict_proba(self, X):
        arr = self._to_array(X)
        return {f: self.models_dict[f].predict_proba(arr)[:, 1] for f in self.flavor_categories}

    def predict(self, X):
        probas = self.predict_proba(X)
        return {f: (probas[f] >= self.thresholds[f]).astype(int) for f in self.flavor_categories}

# Because I didnt add this earlier, I spent an hour debugging the issue, apparently it needed to be injected into __main__ for the inference process to resolve the reference.
sys.modules[__name__].__dict__["CocktailTastePredictor"] = CocktailTastePredictor
if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]
sys.modules["__main__"].CocktailTastePredictor = CocktailTastePredictor


def train(args):
    print("=" * 60)
    print("CocktailAI - Training Job")
    print("=" * 60)

    df_full = pd.read_csv(os.path.join(args.train, "cocktail_dataset.csv"))
    df = df_full[df_full[FLAVOR_CATEGORIES].sum(axis=1) > 0].copy()
    print(f"Loaded {len(df)} recipes with at least one flavor tag")

    # I marked ingriedient columns with _pct suffix.
    ingredient_columns = [c for c in df.columns if c.endswith("_pct")]
    X, y = df[ingredient_columns].copy(), df[FLAVOR_CATEGORIES].copy()

    # Since we had some classes with very few cocktail recipes, I created a rare category for stratification to ensure they are represented in both train and test sets.
    combo = y.astype(str).agg("".join, axis=1)
    rare = combo.value_counts()[lambda s: s < 3].index
    stratify = combo.where(~combo.isin(rare), other="rare")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=stratify
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    trained_models, all_metrics = {}, {}
    for flavor in FLAVOR_CATEGORIES:
        model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            max_iter=1000,
            random_state=SEED,
            class_weight='balanced'
        )
        model.fit(X_train, y_train[flavor])

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= OPTIMAL_THRESHOLDS[flavor]).astype(int)
        metrics = {
            "accuracy": round(accuracy_score(y_test[flavor], y_pred), 4),
            "precision": round(precision_score(y_test[flavor], y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test[flavor], y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test[flavor], y_pred, zero_division=0), 4),
            "auc": round(roc_auc_score(y_test[flavor], y_proba), 4),
        }
        trained_models[flavor] = model
        all_metrics[flavor] = metrics
        print(f"{flavor:<12} AUC={metrics['auc']:.3f}  F1={metrics['f1']:.3f}")

    production_model = CocktailTastePredictor(
        models_dict = trained_models,
        thresholds = OPTIMAL_THRESHOLDS,
        feature_columns = list(ingredient_columns),
        flavor_categories = FLAVOR_CATEGORIES,
    )

    joblib.dump(production_model, os.path.join(args.model_dir, "model.joblib"))

    meta = {
        "feature_columns": list(ingredient_columns),
        "flavor_categories": FLAVOR_CATEGORIES,
        "thresholds": OPTIMAL_THRESHOLDS,
        "test_metrics": all_metrics,
    }
    with open(os.path.join(args.model_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    os.makedirs(args.output_data_dir, exist_ok=True)
    with open(os.path.join(args.output_data_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Training complete - artifacts saved to {args.model_dir}")

# Github Ref link attached in the rpeort for this code block.
def model_fn(model_dir):
    sys.modules["__main__"].CocktailTastePredictor = CocktailTastePredictor
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print(f"[model_fn] Loaded - {len(model.feature_columns)} features, {len(model.flavor_categories)} classifiers")
    return model

def input_fn(request_body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(request_body)
    if "population" not in payload:
        raise ValueError("Request body must contain a 'population' key")
    return np.array(payload["population"], dtype=np.float64)

def predict_fn(population_array, model):
    return model.predict_proba(population_array)

def output_fn(proba_dict, accept="application/json"):
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps({f: p.tolist() for f, p in proba_dict.items()}), "application/json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    train(parser.parse_args())