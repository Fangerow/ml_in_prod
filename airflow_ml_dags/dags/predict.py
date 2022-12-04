from os.path import join
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

from scheduler_configs import airfl_cfg


with DAG(
    "predict",
    default_args=airfl_cfg['default_args'],
    start_date=airfl_cfg['start_date'],
    schedule_interval="@daily",
) as dag:
    start_predict = DummyOperator(task_id='start-predict')

    data_checking = FileSensor(
        task_id="wait-for-data",
        filepath=str(join(airfl_cfg['raw_data'], "data.csv")),
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    make_prediction = DockerOperator(
        image="airflow-predict",
        command=f"--input-dir {airfl_cfg['raw_data']} --output-dir {airfl_cfg['predictions_path']} --model-path {airfl_cfg['model_path']}",
        network_mode="bridge",
        do_xcom_push=False,
        task_id="make-prediction",
        volumes=[f"{airfl_cfg['volumes_path']}:/data"]
    )

    start_predict >> data_checking >> make_prediction
