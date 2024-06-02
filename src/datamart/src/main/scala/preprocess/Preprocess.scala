package preprocess

import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.DataFrame


object Preprocessor {

  def fillNa(df: DataFrame): DataFrame = df.na.fill(0.0)

  def assembleVector(df: DataFrame): DataFrame = {
    val outputCol = "features"
    val inputCols: Array[String] = Array("energy-kcal_100g", "fat_100g", "proteins_100g", "carbohydrates_100g")

    val vector_assembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol(outputCol)
      .setHandleInvalid("skip")
    vector_assembler.transform(df)
  }

  def scaleAssembledDataset(df: DataFrame): DataFrame = {
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaled_features")
    val scalerModel = scaler.fit(df)
    scalerModel.transform(df)
  }
}