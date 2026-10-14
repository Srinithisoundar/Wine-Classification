import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 1. Load dataset
wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)
data["target"] = wine.target  # 0, 1, 2 -> three cultivars

print("Shape:", data.shape)
print("\nClass distribution:\n", data["target"].value_counts())
print("\nMissing values:\n", data.isnull().sum().sum())
print("\nFeature summary:\n", data.describe())

# 2. Exploratory look at class separation (pick 2 informative features)
plt.figure(figsize=(6, 5))
sns.scatterplot(data=data, x="flavanoids", y="color_intensity", hue="target", palette="deep")
plt.title("Wine cultivars by flavanoids vs color intensity")
plt.savefig("wine_eda.png", dpi=100, bbox_inches="tight")
plt.close()

# 3. Train/test split (stratified — mirrors the spam report's reasoning:
#    preserve class proportions across all 3 target classes)
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature scaling — necessary here (unlike Iris) because Wine's
#    features sit on very different numeric scales (alcohol ~11-15,
#    proline ~300-1700). Logistic Regression is scale-sensitive.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model training
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Prediction and evaluation (macro-averaged since it's 3-class)
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print("\n--- Results ---")
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision (macro): {precision*100:.2f}%")
print(f"Recall (macro):    {recall*100:.2f}%")
print(f"F1-score (macro):  {f1*100:.2f}%")
print("\nConfusion matrix:\n", cm)
print("\nFull classification report:\n", classification_report(y_test, y_pred, target_names=wine.target_names))

# 7. Test on a couple of unseen samples (pulled from the test set itself,
#    as stand-ins for "new" samples — mirrors Sanjai's new-message test)
sample_idx = [0, 5]
sample_preds = model.predict(X_test_scaled[sample_idx])
print("\nSample predictions:", sample_preds, "Actual:", y_test.iloc[sample_idx].values)
