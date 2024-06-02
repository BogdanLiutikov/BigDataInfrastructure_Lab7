import findspark
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, current_timestamp

from .datamart import DataMart
from .logger import Logger

findspark.init()


class KMean:
    def __init__(self, datamart: DataMart) -> None:
        self.logger = Logger(True).get_logger(__name__)
        self.datamart = datamart
        self.evaluator = ClusteringEvaluator(metricName='silhouette')

    def train(self, data: DataFrame, k: int = 5):
        self.logger.info("Training... ")
        self.model = KMeans(k=k, seed=42, featuresCol='scaled_features', predictionCol='prediction').fit(data)
        self.logger.info("Done")

    def get_dataframe_fromdict(self, data: list[dict]) -> DataFrame:
        return self.spark.createDataFrame(data, schema=self.schema)

    def predict(self, data: DataFrame):
        self.predictions = self.model.transform(data)
        self.datamart.write_predictions(self.predictions.select("id", "features", "prediction").withColumn("timestamp", current_timestamp()))
        return self.predictions

    def eval(self, predictions):
        silhouette = self.evaluator.evaluate(predictions)
        return silhouette