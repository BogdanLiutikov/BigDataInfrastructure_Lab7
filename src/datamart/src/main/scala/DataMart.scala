import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import db.{DataBase}
import org.apache.spark.sql.types.{DoubleType, StructType}
import preprocess.Preprocessor

class DataMart(spark: SparkSession) {

  private val db = new DataBase(spark)

  def readAndProcessDataset(path: String, sep: String): DataFrame = {
    val df = spark.read
      .format("csv")
      .option("header", "true")
      .option("sep", sep)
      .option("inferSchema", "true")
      .load(path)

    val transforms: Seq[DataFrame => DataFrame] = Seq(
      Preprocessor.fillNa,
      Preprocessor.assembleVector,
      Preprocessor.scaleAssembledDataset
    )

    val transformed = transforms.foldLeft(df) { (df, f) => f(df) }
    transformed
  }

  def write(df: DataFrame, keyspace: String, table: String, mode: String): Unit = {
    db.writeDatababse(df, keyspace, table, mode)
  }

  def read(keyspace: String, table: String): DataFrame = {
    db.readDatabase(keyspace, table)
  }

  def writeData(df: DataFrame): Unit = {
    db.writeDatababse(df, "food", "data", "overwrite")
  }

  def writePredictions(df: DataFrame): Unit = {
    db.writeDatababse(df, "food", "predictions", "append")
  }
}
