import os
import pickle

import findspark
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pyspark import SparkConf
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.functions import col
from configparser import ConfigParser

findspark.init()

class KMean:
    def __init__(self, config_path: str = None) -> None:
        config = ConfigParser()
        config.read('config.ini')
        spark_config = config['spark']
        self.spark: SparkSession = (SparkSession.builder
                                    .master('local')
                                    .appName('KMean')
                                    .config(map={
                                        'spark.driver.cores': spark_config.get('spark.driver.cores'),
                                        'spark.executor.instances': spark_config.get('spark.executor.instances'),
                                        'spark.executor.cores': spark_config.get('spark.executor.cores'),
                                        'spark.executor.memory': spark_config.get('spark.executor.memory'),
                                        'spark.dynamicAllocation.enabled': spark_config.get('spark.dynamicAllocation.enabled'),
                                        # 'spark.log.level': 'ALL'
                                    })
                                    .getOrCreate())

        self.evaluator = ClusteringEvaluator(metricName='silhouette')

    def load_data(self, file_path: str, limit: int = None):
        if os.path.exists('schema.pickle'):
            with open('schema.pickle', 'rb') as s:
                print('readSchema')
                schema = pickle.load(s)
                self.data = self.spark.read.csv(file_path, sep='\t', header=True, schema=schema)
        else:
            with open('schema.pickle', 'wb') as s:
                self.data = self.spark.read.csv(file_path, sep='\t', header=True, inferSchema=True)
                pickle.dump(self.data.schema, s)
        self.data.printSchema()
        if limit:
            self.data = self.data.limit(limit)
        print('Data count: ', self.data.count())
        return self.data

    def make_feature_column(self, data: DataFrame, input_columns=['energy-kcal_100g', 'fat_100g', 'proteins_100g', 'carbohydrates_100g']):
        vs = VectorAssembler(inputCols=input_columns, outputCol='features')
        feature_data = vs.transform(data.fillna(0))
        return feature_data

    def standard_scale(self, train_data: DataFrame, tranform_data: DataFrame = None):
        scaler = StandardScaler(
            inputCol='features',
            outputCol='scaled_features',
            withStd=True,
            withMean=True,
        )

        if tranform_data is None:
            tranform_data = train_data
            
        self.scaler_model = scaler.fit(train_data)
        scaled_data = self.scaler_model.transform(tranform_data)
        return scaled_data

    def train(self, data: DataFrame, k: int = 5):
        self.model = KMeans(k=k, seed=42, featuresCol='features',
                            predictionCol='prediction').fit(data)

    def predict(self, data: DataFrame):
        self.predictions = self.model.transform(data)
        return self.predictions

    def eval(self, predictions):
        silhouette = self.evaluator.evaluate(predictions)
        return silhouette

    def plot(self, predictions: DataFrame):
        pca = PCA(k=2, inputCol='scaled_features', outputCol="pca_features")
        pca_fit = pca.fit(predictions)
        transformed_df = pca_fit.transform(predictions).select("pca_features")
        pandas_df = transformed_df.select(col("pca_features")).toPandas()
        pandas_df['x'] = pandas_df['pca_features'].apply(lambda x: x[0])
        pandas_df['y'] = pandas_df['pca_features'].apply(lambda x: x[1])
        pandas_df = pandas_df.drop(columns=['pca_features'])

        centers_df = self.spark.createDataFrame([(Vectors.dense(c),) for c in self.model.clusterCenters()], ['features'])
        centers_df = self.scaler_model.transform(centers_df)

        centers_pandas_df = pca_fit.transform(centers_df).select("pca_features").toPandas()
        centers_pandas_df['x'] = centers_pandas_df['pca_features'].apply(lambda x: x[0])
        centers_pandas_df['y'] = centers_pandas_df['pca_features'].apply(lambda x: x[1])
        centers_pandas_df = centers_pandas_df.drop(columns=['pca_features'])

        print(centers_pandas_df.head())

        fig, ax = plt.subplots()
        labels = predictions.select('prediction').toPandas().iloc[:, 0]
        colors = ['blue', 'red', 'yellow', 'green', 'pink', 'orange']
        for i, label in enumerate(labels.unique()):
            mask = labels == label
            ax.scatter(pandas_df.loc[mask, 'x'], pandas_df.loc[mask, 'y'], label=label, c=colors[i])
        ax.scatter(centers_pandas_df['x'], centers_pandas_df['y'], marker='x', c='black', label='Centers')
        ax: Axes
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('KMeans Clusters')
        ax.legend()
        plt.show()
