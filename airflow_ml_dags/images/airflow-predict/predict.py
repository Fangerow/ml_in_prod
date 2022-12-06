import joblib
import logging
import click
import pandas as pd
from os.path import join
from os import makedirs

logger = logging.getLogger("predict")


@click.command("predict")
@click.option("--input-dir", required=True)
@click.option("--output-dir", required=True)
@click.option("--model-path", required=True)
def predict(input_dir: str, output_dir: str, model_path: str):
    data = pd.read_csv(join(input_dir, "data.csv"))
    model = joblib.load(model_path)
    predictions = model.predict(data)
    data = pd.DataFrame(predictions, columns=["target"])

    makedirs(output_dir, exist_ok=True)
    data.to_csv(join(output_dir, "predictions.csv"),
                index=False)

    logger.info("Prediction ready")


if __name__ == '__main__':
    predict()
