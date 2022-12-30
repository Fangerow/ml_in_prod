import joblib
import logging
import json
import click
import pandas as pd
from sklearn.metrics import recall_score, precision_score
from os.path import join

logger = logging.getLogger("validate")


@click.command("download")
@click.option("--input-dir", required=True)
@click.option("--models-dir", required=True)
def validate(input_dir: str, models_dir: str):
    logger.info("Model is loading..")

    model = joblib.load(join(models_dir, "model.joblib"))

    logger.info("Model loaded succesfully, start making predictions..")
    val_data = pd.read_csv(join(input_dir, "val.csv"))
    val_labels = val_data[['target']].values
    val_matrix = val_data.drop(columns=['target'])

    predictions = model.predict(val_matrix)
    metrics = {
        "recall": recall_score(val_labels, predictions),
        "precision": precision_score(val_labels, predictions),
    }

    with open(join(models_dir, "metrics.json"), "w") as result_training:
        json.dump(metrics, result_training)
    logger.info("Predictions saved")


if __name__ == "__main__":
    validate()
