import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_wine

def main():
   
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    print("Dataset shape:", X.shape)
    print("Feature names:", iris.feature_names)
    print("Target classes:", iris.target_names)
    print("First five rows:")
    print(X.head())
    print("Target distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nTest accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def main():
        # Load a different built-in dataset
        wine = load_wine()
        X = pd.DataFrame(wine.data, columns=wine.feature_names)
        y = pd.Series(wine.target, name="target")

        print("Dataset shape:", X.shape)
        print("Feature names:", wine.feature_names)
        print("Target classes:", wine.target_names)
        print("First five rows:")
        print(X.head())
        print("Target distribution:")
        print(y.value_counts())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print("\nTest accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=wine.target_names))

        depths = list(range(1, 11))
        test_acc = []
        for d in depths:
            clf = DecisionTreeClassifier(max_depth=d, random_state=42)
            clf.fit(X_train, y_train)
            test_acc.append(accuracy_score(y_test, clf.predict(X_test)))

        plt.figure()
        plt.plot(depths, test_acc, marker="o")
        plt.title("Decision Tree Test Accuracy by max_depth")
        plt.xlabel("max_depth")
        plt.ylabel("Accuracy")
        plt.xticks(depths)
        plt.grid(True)
        plt.show()

    if __name__ == "__main__":
        main()