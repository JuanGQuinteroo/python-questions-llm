import random

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _resolver_pipeline_clasificacion_balanceada(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred),
    }


def generar_caso_de_uso_pipeline_clasificacion_balanceada():
    n_muestras = random.randint(180, 380)
    n_features = random.randint(6, 12)
    n_informative = random.randint(2, n_features - 2)

    peso_clase_1 = random.uniform(0.12, 0.32)
    random_state_data = random.randint(0, 5000)

    X, y = make_classification(
        n_samples=n_muestras,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        weights=[1.0 - peso_clase_1, peso_clase_1],
        flip_y=random.uniform(0.0, 0.03),
        class_sep=random.uniform(0.7, 1.8),
        random_state=random_state_data,
    )

    test_size = random.choice([0.20, 0.25, 0.30])
    random_state = random.randint(0, 5000)

    input_data = {
        "X": X.copy(),
        "y": y.copy(),
        "test_size": test_size,
        "random_state": random_state,
    }

    output_data = _resolver_pipeline_clasificacion_balanceada(
        X, y, test_size, random_state
    )
    return input_data, output_data
