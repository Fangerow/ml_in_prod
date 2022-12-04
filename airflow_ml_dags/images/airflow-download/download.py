import logging
import click
import numpy as np
import pandas as pd
from os.path import join
from os import makedirs
from numpy.random import normal, binomial, choice, randint, poisson

logger = logging.getLogger("download")


@click.command("download")
@click.option("--output-dir", required=True)
def download(output_dir: str):
    size = randint(100, 200)
    logger.info(f"Start generating with data size {size}")

    data = pd.DataFrame()
    data["age"] = normal(loc=55, scale=10, size=size)
    data["sex"] = binomial(n=1, p=0.7, size=size)
    data["cp"] = randint(low=0, high=4, size=size)
    data["trestbps"] = normal(loc=130, scale=20, size=size)
    data["chol"] = normal(loc=250, scale=50, size=size)
    data["fbs"] = binomial(n=1, p=0.2, size=size)
    data["restecg"] = choice([0, 1, 2], size=size, p=[0.48, 0.48, 0.04])
    data["thalach"] = normal(loc=150, scale=20, size=size)
    data["exang"] = binomial(n=1, p=0.33, size=size)
    data["oldpeak"] = np.clip(normal(loc=1, scale=2, size=size), a_min=0, a_max=None).round(1)
    data["slope"] = choice([0, 1, 2], size=size, p=[0.08, 0.46, 0.46])
    data["ca"] = poisson(lam=0.5, size=size)
    data["thal"] = choice([0, 1, 2, 3], size=size, p=[0.01, 0.07, 0.52, 0.4])
    target = pd.Series(binomial(n=1, p=0.55, size=size))
    data = data.astype(np.int64)

    makedirs(output_dir, exist_ok=True)
    data.to_csv(join(output_dir, "data.csv"),
                index=False)
    target.to_csv(join(output_dir, "target.csv"),
                  index=False)

    logger.info("data downloading completed")


if __name__ == '__main__':
    download()
