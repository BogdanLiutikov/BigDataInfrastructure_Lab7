from pyspark.sql import SparkSession
from pyspark.sql import functions as sf

spark: SparkSession = SparkSession.builder.master("local").appName("Word Count").getOrCreate()

if __name__ == "__main__":
    textFile = spark.read.text("word_count_input.txt")
    wordCounts = textFile.select(sf.explode(sf.split(textFile.value, "\s+")).alias("word")).groupBy("word").count().collect()
    print(wordCounts)