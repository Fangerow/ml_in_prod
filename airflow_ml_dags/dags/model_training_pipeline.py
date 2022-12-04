from os.path import join
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

from scheduler_configs import airfl_cfg


with DAG(
    "model_training_pipeline",
    default_args=airfl_cfg['default_args'],
    start_date=airfl_cfg['start_date'],
    default_view="graph",
    schedule_interval="@weekly",
) as dag:
    start_pipeline = DummyOperator(task_id='start-pipeline')

    design_matrix_checking = FileSensor(
        task_id="wait-for-data",
        filepath=str(join(airfl_cfg['raw_data'], "data.csv")),
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    labels_checking = FileSensor(
        task_id="wait-for-target",
        filepath=str(join(airfl_cfg['raw_data'], "target.csv")),
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    data_preprocessing = DockerOperator(
        image="train_validation",
        command=f"--input-dir {airfl_cfg['raw_data']} --output-dir {airfl_cfg['precessed_data_path']}",
        network_mode="bridge",
        do_xcom_push=False,
        task_id="data-preprocessing",
        volumes=[f"{airfl_cfg['volumes_path']}:/data"],
        entrypoint="python preprocess.py"
    )

    data_split = DockerOperator(
        image="train_validation",
        command=f"--input-dir {airfl_cfg['precessed_data_path']} --val-size {airfl_cfg['size']}",
        network_mode="bridge",
        do_xcom_push=False,
        task_id="data-split",
        volumes=[f"{airfl_cfg['volumes_path']}:/data"],
        entrypoint="python split.py"
    )

    model_training = DockerOperator(
        image="train_validation",
        command=f"--input-dir {airfl_cfg['precessed_data_path']} --models-dir {airfl_cfg['pretrained_models']}",
        network_mode="bridge",
        do_xcom_push=False,
        task_id="model-training",
        volumes=[f"{airfl_cfg['volumes_path']}:/data"],
        entrypoint="python train.py"
    )

    model_validation = DockerOperator(
        image="train_validation",
        command=f"--input-dir {airfl_cfg['precessed_data_path']} --models-dir {airfl_cfg['pretrained_models']}",
        network_mode="bridge",
        do_xcom_push=False,
        task_id="model-validation",
        volumes=[f"{airfl_cfg['volumes_path']}:/data"],
        entrypoint="python validate.py"
    )

    start_pipeline >> [design_matrix_checking, labels_checking] >> data_preprocessing >> \
    data_split >> model_training >> model_validation
