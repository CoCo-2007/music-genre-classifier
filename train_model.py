import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the feature table created by extract_features.py
df = pd.read_csv("features.csv")
print("Dataset shape:", df.shape)

# 2. Separate features (X) and genre label (y)
X = df.drop("genre", axis=1)
y = df["genre"]

# 3. Split: 80% training, 20% testing (same genre proportions in both)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Scale features. Fit ONLY on training data, then apply to test data.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. SVM (uses scaled features)
print("\nTraining SVM...")
svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")
print(classification_report(y_test, svm_predictions))

# 6. Random Forest (uses original features)
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
print(classification_report(y_test, rf_predictions))

# 7. Pick the better model
if svm_accuracy >= rf_accuracy:
    best_model, best_name = svm_model, "SVM"
else:
    best_model, best_name = rf_model, "Random Forest"

print("--------------------------------")
print("Best model:", best_name)
print("--------------------------------")

# 8. Save model and scaler
joblib.dump(best_model, "models/genre_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Saved: models/genre_model.pkl and models/scaler.pkl")