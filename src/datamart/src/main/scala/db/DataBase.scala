package db

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import com.datastax.spark.connector._

class DataBase(spark: SparkSession) {
  def writeDatababse(
      df: DataFrame,
      keyspace: String,
      table: String,
      mode: String
  ): Unit = {
    try {
      df.write
        .format("org.apache.spark.sql.cassandra")
        .options(Map("keyspace" -> keyspace, "table" -> table, "confirm.truncate" -> "true"))
        .mode(mode)
        .save()
      println("Data written to Cassandra successfully")
    } catch {
      case e: Exception =>
        println("Failed to write to Cassandra")
        e.printStackTrace()
    }
  }

  def readDatabase(
      keyspace: String,
      table: String
  ): DataFrame = {
    spark.read
      .format("org.apache.spark.sql.cassandra")
      .options(Map("keyspace" -> keyspace, "table" -> table))
      .load()
  }
}
