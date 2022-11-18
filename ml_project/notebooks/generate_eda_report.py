import sweetviz as sv
import pandas as pd
from os.path import exists


def make_report(csv):
    """
    :param csv: path to csv dataset, that should be analyzed automatically
    """
    assert exists(csv)

    data = pd.read_csv(csv)
    report = sv.analyze(data,
                        pairwise_analysis="on")
    report.show_html('common analysis.html')


if __name__ == "__main__":
    make_report('heart_cleveland_upload.csv')
