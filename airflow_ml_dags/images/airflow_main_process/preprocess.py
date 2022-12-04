import logging
import click
import pandas as pd
from os.path import join
from os import makedirs

logger = logging.getLogger("preprocess")


@click.command()
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir: str):
    data = pd.read_csv(join(input_dir, "data.csv"))
    target = pd.read_csv(join(input_dir, "target.csv")).set_axis(["target"], axis=1)

    makedirs(output_dir, exist_ok=True)

    data_processed = pd.concat([data, target], axis=1)
    data_processed.to_csv(join(output_dir, "train_data.csv"),
                          index=False)

    logger.info("Preprocessing completed")


if __name__ == '__main__':
    preprocess()
