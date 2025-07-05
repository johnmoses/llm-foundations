import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

def main():
    # Sample dataset - replace with your real dataset
    data = {
        "symptoms": [
            "fever;cough;headache",
            "chest pain;shortness of breath",
            "nausea;headache;dizziness",
            "fever;cough",
            "fatigue;weight loss",
        ],
        "disease": [
            "common cold",
            "heart attack",
            "migraine",
            "flu",
            "diabetes",
        ]
    }

    df = pd.DataFrame(data)

    # Vectorize symptoms (split by semicolon)
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(";"))
    X = vectorizer.fit_transform(df["symptoms"])
    y = df["disease"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Decision Tree classifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save model and vectorizer
    joblib.dump(model, "disease_predictor.joblib")
    joblib.dump(vectorizer, "symptom_vectorizer.joblib")
    print("Model and vectorizer saved.")

if __name__ == "__main__":
    main()
