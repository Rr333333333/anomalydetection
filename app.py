from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import io, base64
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

app = Flask(__name__)

# ---------------- SYNTHETIC DATA ----------------
def generate_synthetic_data(n=5000):
    np.random.seed(42)
    df = pd.DataFrame({
        "duration": np.random.randint(0, 1000, n),
        "src_bytes": np.random.randint(0, 10000, n),
        "dst_bytes": np.random.randint(0, 15000, n),
        "hot": np.random.randint(0, 5, n),
        "failed_logins": np.random.randint(0, 3, n),
        "urgent": np.random.randint(0, 2, n),
        "wrong_frag": np.random.randint(0, 2, n),
        "label": np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    })
    df["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
    return df

# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template('index.html')

# ---------------- TRAIN ----------------
@app.route('/train', methods=['POST'])
def train_model():
    file = request.files.get('file')
    if file:
        data = pd.read_csv(file)
    else:
        data = generate_synthetic_data()

    data = data.select_dtypes(include=['number']).dropna()
    if 'label' not in data.columns:
        return jsonify({'error': "No 'label' column found"}), 400

    X = data.drop('label', axis=1)
    y = data['label']

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)

    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_res)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y_res, test_size=0.3, random_state=42)

    # ---------------- Models ----------------
    models = {
        "Isolation Forest": IsolationForest(contamination=0.15),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel='rbf', C=10, gamma=0.1, probability=True),
        "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
        "LightGBM": LGBMClassifier()
    }

    metrics = {}
    preds_combined = []

    for name, model in models.items():
        if name == "Isolation Forest":
            model.fit(X_train)
            preds = model.predict(X_test)
            preds = np.where(preds == -1, 1, 0)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        acc = round(accuracy_score(y_test, preds) * 100, 2)
        prec = round(precision_score(y_test, preds) * 100, 2)
        rec = round(recall_score(y_test, preds) * 100, 2)
        metrics[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec}
        preds_combined.append(preds)

    # Ensemble
    ensemble_pred = (np.sum(preds_combined, axis=0) >= 3).astype(int)
    metrics["Ensemble"] = {
        "Accuracy": round(accuracy_score(y_test, ensemble_pred) * 100, 2),
        "Precision": round(precision_score(y_test, ensemble_pred) * 100, 2),
        "Recall": round(recall_score(y_test, ensemble_pred) * 100, 2)
    }

    # ---------------- Chart ----------------
    counts = pd.Series(ensemble_pred).value_counts()
    colors = ['green' if cls == 0 else 'red' for cls in counts.index]
    plt.figure(figsize=(5,4))
    plt.bar(counts.index.astype(str), counts.values, color=colors)
    plt.title("Anomaly Distribution (0=Normal, 1=Anomaly)")
    plt.ylabel("Count")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return render_template('results.html', metrics=metrics, chart=chart_base64)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

