import mlflow
import os
import pandas as pd

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Прочитаем данные
housing = fetch_california_housing(as_frame=True)

# train val test
X_train, X_test, y_train, y_test = train_test_split(housing['data'], housing['target'])
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)


# Предобработка данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Установка эксперимента
experiment_name = "Alexander_Kuchiev"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(name = experiment_name, artifact_location = "s3://test-bucket-mlops-akuchiev/mlflow")
else:
    mlflow.set_experiment(experiment_name)


models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(),
    "DecisionTree": DecisionTreeRegressor()
}


# Создадим parent run.
with mlflow.start_run(run_name="ascetto", description = "parent") as parent_run:
    for model_name in models.keys():
        # Запустим child run на каждую модель.
        with mlflow.start_run(run_name=model_name, nested=True) as child_run:
            model = models[model_name]
            
            # Обучим модель.
            model.fit(pd.DataFrame(X_train), y_train)
        
            # Сделаем предсказание.
            prediction = model.predict(X_val)

            # Создадим валидационный датасет.
            eval_df = X_val.copy()
            eval_df["target"] = y_val
        
            # Сохраним результаты обучения с помощью MLFlow.
            signature = infer_signature(X_test, prediction)
            model_info = mlflow.sklearn.log_model(model, "linreg", signature=signature, 
                                                  registered_model_name=f"sk-learn-{model_name}-reg-model")
            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets="target",
                model_type="regressor",
                evaluators=["default"],
            )

