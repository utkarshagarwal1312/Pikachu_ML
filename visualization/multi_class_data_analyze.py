import logging
import math
import warnings
import matplotlib
import pandas as pd
import seaborn as sns
import numpy

from matplotlib import pyplot as plt

matplotlib.use("TkAgg")
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LINE_SIZE = 90


class MultiClassDataAnalyze:
    # Data Visualization for Multi Class Classification
    # Features : Data Description, Head, Dimensions, Feature Histogram Plot, Feature Density Plot
    # Assumptions : Class Feature in CSV is labeled as Outcome, Column names are Camel Cased non space separated
    def __init__(self, csv_path):
        try:
            self.df = pd.read_csv(csv_path)
            self.features = self.df.columns[:-1]
            self.classes = self.df.Outcome.unique()
        except Exception as e:
            print(e)
            logger.error("Error creating dataset")

    @staticmethod
    def _divider():
        logger.info("".join(['*'] * LINE_SIZE))

    def data_info(self):
        self._divider()
        logger.info(" Head of Dataset")
        self._divider()
        logger.info(self.df.head())
        self._divider()
        logger.info(" Dataset Describe")
        self._divider()
        logger.info(self.df.describe())

    def histogram(self):
        self.df.hist()
        plt.tight_layout()
        plt.show()

    def density_plots(self):
        matrix_dim = int(math.ceil(math.sqrt(len(self.features))))
        plt.subplots(matrix_dim, matrix_dim, figsize=(20, 20))
        for idx, col in enumerate(self.features):
            ax = plt.subplot(matrix_dim, matrix_dim, idx + 1)
            ax.yaxis.set_ticklabels([])
            for outcome in self.classes:
                sns.distplot(self.df.loc[self.df.Outcome == outcome][col], hist=False, axlabel=False,
                             kde_kws={'linestyle': '--', 'color': numpy.random.rand(3, ),
                                      'label': "Class : {}".format(str(outcome))})
            ax.set_title(col)
        plt.tight_layout()
        plt.show()

    def execute(self):
        self.data_info()
        self.histogram()
        self.density_plots()

# Sample Usage
# MultiClassDataAnalyze("/home/utkarsh/PycharmProjects/Pikachu_ML/neuralnets/resources/iris.csv").execute()
