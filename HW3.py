import mlflow
import os
import io
import json
import pickle
import pandas as pd
from datetime import datetime, timedelta

from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from typing import Any, Dict, Literal
# YOUR IMPORTS HERE


BUCKET = Variable.get("S3_BUCKET")

DEFAULT_ARGS = {
    'owner': 'Alexander Kuchiev',
    'retry': 3,
    'retry_delay': timedelta(minutes=1)
}
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET = "MedHouseVal"




dag = DAG(
    "Alexander_Kuchiev",
    default_args=DEFAULT_ARGS,
    schedule_interval = '0 1 * * *',
    start_date = days_ago(2),
    catchup = False,
    tags = ['mlops']
)

# models
model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))



def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)


def init(**kwargs) -> Dict[str, Any]:
    
    configure_mlflow()
    
    metrics = {}
    metrics['pipeline_start_dttm'] = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
    




    experiment_name = "Alexander_Kuchiev_"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Создаем или выбираем эксперимент
    if experiment:
        experiment_id = mlflow.set_experiment(experiment_name).experiment_id
        
    else:
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location = "s3://test-bucket-mlops-akuchiev/AlexKuchiev/mlflow")
        

    # Запускаем родительский run
    with mlflow.start_run(experiment_id = experiment_id, run_name="ascetto", description="parent") as parent_run:
        # Сохраняем experiment_id и run_id в словарь для передачи между шагами

        metrics['experiment_id'] = experiment_id
        metrics['run_id'] = parent_run.info.run_id
        #print('METRICS CHECK_INIT')
        #print(metrics)
    return metrics

def get_data(**kwargs) -> Dict[str, Any]:
        # YOUR CODE HERE
        start = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
        s3_hook = S3Hook("s3_connection")
        
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids = 'init')
        
        
        # fetch data
        california_housing = fetch_california_housing()
        df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
        df['MedHouseVal'] = california_housing.target

        buffer = io.BytesIO()
        df.to_pickle(buffer)
        buffer.seek(0)

        s3_hook.load_file_obj(
            file_obj = buffer, 
            key = f'AlexKuchiev/datasets/california_housing.pkl', 
            bucket_name = BUCKET, 
            replace = True
        )

        end = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
        metrics['get_data_start_dttm'] = start
        metrics['get_data_end_dttm'] = end
        metrics['data_size'] = df.shape

        return metrics


def prepare_data(**kwargs) -> Dict[str, Any]:
   
        # YOUR CODE HERE
        start = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
        s3_hook = S3Hook("s3_connection")
        
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids = 'get_data')
        print('METRICS CHECK_prepDATA')
        print(metrics)

        file = s3_hook.download_file(key=f'AlexKuchiev/datasets/california_housing.pkl', bucket_name=BUCKET)
        data = pd.read_pickle(file)

        # Сделать препроцессинг
        # Разделить на фичи и таргет
        X, y = data[FEATURES], data[TARGET]

        # Разделить данные на обучение и тест и валидацию
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
       )
        
        #X_val, X_test, y_val, y_test = train_test_split(
        #    X_test, y_test, test_size=0.5, random_state=42
        #)
        
        # Обучить стандартизатор на train
        scaler = StandardScaler()
        X_train_fitted = scaler.fit_transform(X_train)
        #X_val_fitted = scaler.transform(X_val)
        X_test_fitted = scaler.transform(X_test)

        for name, data in zip(["X_train", "X_test", "y_train", "y_test"], [X_train_fitted, X_test_fitted, y_train, y_test]):
            filebuffer = io.BytesIO()
            pickle.dump(data, filebuffer)
            filebuffer.seek(0)
            s3_hook.load_file_obj(
                file_obj=filebuffer,
                key=f"AlexKuchiev/datasets/{name}.pkl",
                bucket_name=BUCKET,
                replace=True,
            )
        end = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
        
        metrics['prepare_data_start_dttm'] = start
        metrics['prepare_data_end_dttm'] = end
        metrics['model_features'] = list(X.columns)

        return metrics    


def train_model(model_name, **kwargs) -> Dict[str, Any]:
    # YOUR CODE HERE
        configure_mlflow()
        start = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
        s3_hook = S3Hook("s3_connection")
        
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids = 'prepare_data')
        
        experiment_id = metrics['experiment_id']
        run_id = metrics['run_id']
        
        print('METRICS CHECK_train')
        print(metrics)

        if not experiment_id or not run_id:
            raise ValueError("Experiment ID or Run ID not found in XCom. Make sure init task executed correctly.")
    
        
        data = {}
        model = models[model_name]
        for name in ["X_train", "X_test", "y_train", "y_test"]:
            file = s3_hook.download_file(
                key=f"AlexKuchiev/datasets/{name}.pkl",
                bucket_name=BUCKET,
            )
            data[name] = pd.read_pickle(file)

        
        
        X_train = pd.DataFrame(data['X_train'], columns = FEATURES)
        X_test = pd.DataFrame(data['X_test'], columns = FEATURES)
        y_train = pd.Series(data['y_train'])
        y_test = pd.Series(data['y_test'])

        
 #with mlflow.start_run(run_name=model_name, nested=True) as child_run:
 #with mlflow.start_run(experiment_id=experiment_id, run_id=run_id, nested=True, run_name=model_name):   
        with mlflow.start_run(experiment_id = experiment_id, parent_run_id = run_id, nested=True, run_name=model_name, description = 'child') as child_run:
            model = models[model_name]

            # Обучим модель.
            #model.fit(pd.DataFrame(X_train), y_train)
            model.fit(X_train, y_train)
        
            # Сделаем предсказание.
            #prediction = model.predict(X_val)
            prediction = model.predict(X_test)

            # Создадим валидационный датасет.
            eval_df = X_test.copy()
            eval_df["target"] = y_test
            eval_df.target.fillna(y_train.median(), inplace = True)

                        

            
            # Сохраним результаты обучения с помощью MLFlow.
            signature = infer_signature(X_test, prediction)
            model_info = mlflow.sklearn.log_model(model, "regression", signature=signature, 
                                                  registered_model_name=f"sk-learn-{model_name}-reg-model")
            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets="target",
                model_type="regressor",
                evaluators=["default"],
            )

        end = datetime.now().strftime('%Y.%m.%d %H:%M:%S')

        metrics[f'{model}_train_start_dttm'] = start
        metrics[f'{model}_train_end_dttm'] = end
        
        
        return metrics


def save_results(**kwargs) -> None:
        # YOUR CODE HERE

        s3_hook = S3Hook("s3_connection")
        
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids = 'train_model')
        

        
        buffer = io.BytesIO()
        buffer.write(json.dumps(metrics).encode())
        buffer.seek(0)
    
        s3_hook.load_file_obj(
            file_obj = buffer, 
            key = f"AlexKuchiev/results/metrics.json", 
            bucket_name = BUCKET, 
            replace = True
        )

task_init = PythonOperator(
    task_id = "init",
    python_callable = init,
    provide_context = True,  
    dag=dag,
)

task_get_data = PythonOperator(
    task_id="get_data",
    python_callable=get_data,
    provide_context=True,  
    dag=dag,
)

task_prepare_data = PythonOperator(
    task_id="prepare_data",
    python_callable=prepare_data,
    provide_context=True,  # Передача контекста
    dag=dag,
)

training_model_tasks = []
for model_name in model_names:
    task = PythonOperator(
        task_id=f"train_model_{model_name}",
        python_callable=train_model,
        op_kwargs={"model_name": model_name},
        provide_context=True,  
        dag=dag,
    )
    training_model_tasks.append(task)



task_save_results = PythonOperator(
    task_id="save_results",
    python_callable=save_results,
    provide_context=True,  
    dag=dag,
)

task_init >> task_get_data >> task_prepare_data >> training_model_tasks >> task_save_results