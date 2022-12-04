from datetime import timedelta
from airflow.models import Variable
from airflow.utils.dates import days_ago


airfl_cfg = {
    'default_args': {
        'owner': 'airflow',
        'email': ['airflow@example.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 3,
        'retry_delay': timedelta(minutes=1),
    },
    'start_date': days_ago(7),
    'raw_data': "/data/raw/{{ ds }}",
    'precessed_data_path': "/data/processed/{{ ds }}",
    'pretrained_models': "/data/models/{{ ds }}",
    'predictions_path': "/data/predictions/{{ ds }}",
    'model_path': Variable.get("model_path"),
    'size': 0.2,
    'volumes_path': "/Users/User/Desktop/airlows/oortur/airflow_ml_dags/data",
}
