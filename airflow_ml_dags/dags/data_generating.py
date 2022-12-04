from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from scheduler_configs import airfl_cfg


with DAG(
    "data_generating",
    default_args=airfl_cfg['default_args'],
    start_date=airfl_cfg['start_date'],
    schedule_interval="@daily",
) as dag:
    start_download = DummyOperator(task_id="start-download")
    data_download = DockerOperator(
        image="data_generating",
        command=f"--output-dir {airfl_cfg['raw_data']}",
        network_mode="bridge",
        do_xcom_push=False,
        task_id="download-data",
        volumes=[f"{airfl_cfg['volumes_path']}:/data"],
    )

    start_download >> data_download
