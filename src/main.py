import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp

from .datamart import DataMart
from .kmean import KMean
from .logger import Logger

findspark.init()
logger = Logger(True).get_logger(__name__)

spark = (SparkSession.builder
         .master('local')
         .appName('KMean')
         .config(map={
             "spark.jars.packages": f"com.datastax.spark:spark-cassandra-connector_2.12:3.5.0",
             "spark.jars": f"src/datamart/target/scala-2.12/datamart_2.12-0.1.jar",
             "spark.cassandra.connection.host": "localhost",
             "spark.cassandra.connection.port": "9042"
         })
         .getOrCreate())


data_mart = DataMart(spark)

assembled_data = data_mart.read_dataset(path="data/example.csv", sep=",")
assembled_data.select('id', 'energy-kcal_100g', "fat_100g",
                      "proteins_100g", "carbohydrates_100g", 'features').show(5)
data_mart.write_data(assembled_data.select('id', 'features'))

kmean = KMean(data_mart)
kmean.train(assembled_data)

predict = kmean.predict(assembled_data)
predict.select('id', 'features', 'prediction').show(5)
