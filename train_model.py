import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("Mall_Customers.csv", sep="\t")

# Show column names
print(df.columns)

# Rename columns if needed
df.columns = df.columns.str.strip()

# Convert Gender column
if "Genre" in df.columns:
    df["Genre"] = df["Genre"].map({"Male": 1, "Female": 0})

# Convert Spending Score to numeric
df["Spending_Score"] = pd.to_numeric(
    df["Spending_Score"],
    errors="coerce"
)

# Create target column
df["Segment"] = (df["Spending_Score"] >= 50).astype(int)

# Features and target
X = df[["Genre", "Age", "Annual_Income_(k$)", "Spending_Score"]]
y = df["Segment"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))

# Save model
joblib.dump(model, "model.pkl")

print("Model saved successfully!")