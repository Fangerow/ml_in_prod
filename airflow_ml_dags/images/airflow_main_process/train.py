import joblib
import logging
import click
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from os.path import join
from os import makedirs

logger = logging.getLogger("train")


@click.command("download")
@click.option("--input-dir", required=True)
@click.option("--models-dir", required=True)
def train(input_dir: str, models_dir: str):
    logger.info('Starting training the model')

    train_data = pd.read_csv(join(input_dir, "train.csv"))
    train_labels = train_data[['target']]
    train_matrix = train_data.drop(columns=['target'])

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_matrix, train_labels)

    makedirs(models_dir, exist_ok=True)
    joblib.dump(model, join(models_dir, "model.joblib"))

    logger.info("Training ended Successfully")


if __name__ == '__main__':
    train()
