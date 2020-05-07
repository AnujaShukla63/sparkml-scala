package linreg

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GeneralizedLinearRegression, GeneralizedLinearRegressionModel}
import org.apache.spark.sql.SparkSession

object LinearRegression {

  def main(args: Array[String]): Unit = {

    println("======= Spark ML Pipeline in Scala ========")
    println(" ========= Using Linear Regression ========")

    val trueVal = true

    val spark = SparkSession
      .builder
      .appName("linregapp")
      .config("spark.master", "local")
      .getOrCreate()

    println("======= Created Spark Context ========")

    val dataset = spark.read.format("csv")
      .option("header", trueVal)
      .option("inferSchema", trueVal)
      .load("./data/CensusCanada2016.csv")
      .withColumnRenamed("Median Household Income (Current Year $)","label")

    println("======== Dataset ==========")
    dataset.show(5)

    // Assembling all predictors
    val assembler = new VectorAssembler()
      .setInputCols(dataset.columns.slice(1,12))
      .setOutputCol("features")

    println(" ======= Model Fitting ========")

    val glr = new GeneralizedLinearRegression()
      // Distribution
      .setFamily("gaussian")
      // Relationship/Mapping
      .setLink("identity")
      .setMaxIter(100)

    // Creating pipeline
    val pipeline = new Pipeline().setStages(Array(assembler,glr))

    // Fitting the model
    val lrModel = pipeline.fit(dataset)

    println(" ============  Model Summary =========")

    val estimator =   lrModel.stages(1).asInstanceOf[GeneralizedLinearRegressionModel]

    val summary = estimator.summary

    println(s"Coefficient Standard Errors: ${summary.coefficientStandardErrors.mkString(",")}")
    println(s"P Values: ${summary.pValues.mkString(",")}")
    println(s"Dispersion: ${summary.dispersion}")
    println(s"Null Deviance: ${summary.nullDeviance}")
    println("Deviance Residuals: ")
    summary.residuals().show()

    //Stop spark context
    spark.stop()

  }

}
