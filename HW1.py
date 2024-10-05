from airflow.models import DAG, Variable
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

from sklearn.datasets import fetch_california_housing
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from typing import Any, Dict, Literal
from datetime import datetime, timedelta
import io
import json
import pandas as pd
import pickle


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


model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))


def create_dag(dag_id: str, m_name: Literal["random_forest", "linear_regression", "desicion_tree"]):

    ####### DAG STEPS #######

    def init(m_name: Literal["random_forest", "linear_regression", "desicion_tree"]) -> Dict[str, Any]:
        metrics = {}
        metrics['model_name'] = m_name
        metrics['pipeline_start_dttm'] = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
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
            key = f'AlexKuchiev/{metrics["model_name"]}/datasets/california_housing.pkl', 
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

        file = s3_hook.download_file(key=f'AlexKuchiev/{metrics["model_name"]}/datasets/california_housing.pkl', bucket_name=BUCKET)
        data = pd.read_pickle(file)

        # Сделать препроцессинг
        # Разделить на фичи и таргет
        X, y = data[FEATURES], data[TARGET]

        # Разделить данные на обучение и тест
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
       )

        # Обучить стандартизатор на train
        scaler = StandardScaler()
        X_train_fitted = scaler.fit_transform(X_train)
        X_test_fitted = scaler.transform(X_test)

        for name, data in zip(["X_train", "X_test", "y_train", "y_test"], [X_train_fitted, X_test_fitted, y_train, y_test]):
            filebuffer = io.BytesIO()
            pickle.dump(data, filebuffer)
            filebuffer.seek(0)
            s3_hook.load_file_obj(
                file_obj=filebuffer,
                key=f"AlexKuchiev/{metrics['model_name']}/datasets/{name}.pkl",
                bucket_name=BUCKET,
                replace=True,
            )
        end = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
        
        metrics['prepare_data_start_dttm'] = start
        metrics['prepare_data_end_dttm'] = end
        metrics['model_features'] = list(X.columns)

        return metrics

    def train_model(**kwargs) -> Dict[str, Any]:
        # YOUR CODE HERE

        start = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
        s3_hook = S3Hook("s3_connection")
        
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids = 'prepare_data')

        data = {}
        for name in ["X_train", "X_test", "y_train", "y_test"]:
            file = s3_hook.download_file(
                key=f"AlexKuchiev/{metrics['model_name']}/datasets/{name}.pkl",
                bucket_name=BUCKET,
            )
            data[name] = pd.read_pickle(file)

        # Обучить модель
        model = models[metrics['model_name']]
        model.fit(data["X_train"], data["y_train"])
        prediction = model.predict(data["X_test"])

        result = {}
        result["r2_score"] = r2_score(data["y_test"], prediction)
        result["rmse"] = mean_squared_error(data["y_test"], prediction) ** 0.5
        result["mae"] = median_absolute_error(data["y_test"], prediction)

        end = datetime.now().strftime('%Y.%m.%d %H:%M:%S')

        metrics['train_model_start_dttm'] = start
        metrics['train_model_end_dttm'] = end
        metrics['metrics'] = result
        
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
            key = f"AlexKuchiev/{metrics['model_name']}/results/metrics.json", 
            bucket_name = BUCKET, 
            replace = True
        )
    ####### INIT DAG #######

    dag = DAG(
        dag_id = dag_id,
        schedule_interval = '0 1 * * *',
        start_date = days_ago(2),
        catchup = False,
        tags = ['mlops'],
        default_args = DEFAULT_ARGS
    )

    with dag:
        # YOUR TASKS HERE
        task_init = PythonOperator(task_id = 'init', python_callable = init, dag=dag, op_args=[m_name])

        task_get_data = PythonOperator(task_id = 'get_data', python_callable = get_data, dag = dag, provide_context = True)

        task_prepare_data = PythonOperator(task_id = 'prepare_data', python_callable = prepare_data, dag = dag, provide_context = True)

        task_train_model = PythonOperator(task_id = 'train_model', python_callable = train_model, dag = dag, provide_context = True)

        task_save_results = PythonOperator(task_id = 'save_results', python_callable = save_results, dag = dag, provide_context = True)

        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results


for model_name in models.keys():
    create_dag(f"Alexander_Kuchiev_{model_name}", model_name)