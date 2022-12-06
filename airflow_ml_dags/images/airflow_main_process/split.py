import logging
import click
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join

logger = logging.getLogger("split")


@click.command()
@click.option("--input-dir")
def split(input_dir: str):
    data = pd.read_csv(join(input_dir, "train_data.csv"))
    train, val = train_test_split(data, test_size=0.15)

    train.to_csv(join(input_dir, "train.csv"),
                 index=False)
    val.to_csv(join(input_dir, "val.csv"),
               index=False)

    logger.info("Train/validation splitting completed")


if __name__ == '__main__':
    split()
