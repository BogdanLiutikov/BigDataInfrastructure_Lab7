from pyspark.sql import DataFrame, SparkSession, SQLContext


class DataMart:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.spark_context = spark.sparkContext
        self.sql_context = SQLContext(self.spark_context, spark)
        self.jwm_datamart = self.spark_context._jvm.DataMart(spark._jsparkSession)

    def read_dataset(self, path: str, sep: str = ",") -> DataFrame:
        jvm_data = self.jwm_datamart.readAndProcessDataset(path, sep)
        return DataFrame(jvm_data, self.sql_context)

    def write_predictions(self, df: DataFrame):
        self.jwm_datamart.writePredictions(df._jdf)

    def read_database(self, keyspace: str, table: str) -> DataFrame:
        jvm_data = self.jwm_datamart.read(keyspace, table)
        return DataFrame(jvm_data, self.sql_context)
    
    def write_data(self, data: DataFrame):
        self.jwm_datamart.writeData(data._jdf)