/** */
package com.virtualpairprogrammers;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/** @author uchil */
public class GymCompetitors {

  /** @param args */
  public static void main(String[] args) {

    LoggerContext context = (LoggerContext) LogManager.getContext(false);
    Configuration logConfiguration = context.getConfiguration();
    LoggerConfig loggerConfig = logConfiguration.getLoggerConfig("org.apache");
    loggerConfig.setLevel(Level.WARN);
    context.updateLoggers();

    SparkSession spark =
        SparkSession.builder()
            .appName("GymCompetitors")
            .config("spark.sql.warehouse.dir", "file:///c:/tmp/")
            .master("local[*]")
            .getOrCreate();

    Dataset<Row> csvData =
        spark
            .read()
            .option("header", true)
            .option("inferSchema", true)
            .csv("src/main/resources/GymCompetition.csv");

    csvData.printSchema();

    StringIndexer genderIndexer = new StringIndexer();
    genderIndexer.setInputCol("Gender");
    genderIndexer.setOutputCol("GenderIndex");
    csvData = genderIndexer.fit(csvData).transform(csvData);
    csvData.show();

    OneHotEncoder genderEncoder = new OneHotEncoder();
    genderEncoder.setInputCols(new String[] {"GenderIndex"});
    genderEncoder.setOutputCols(new String[] {"GenderVector"});
    csvData = genderEncoder.fit(csvData).transform(csvData);
    csvData.show();

    VectorAssembler vectorAssembler = new VectorAssembler();
    vectorAssembler.setInputCols(new String[] {"Age", "Height", "Weight", "GenderVector"});
    vectorAssembler.setOutputCol("features");
    Dataset<Row> csvDataWithFeatures = vectorAssembler.transform(csvData);
    Dataset<Row> modelInputData =
        csvDataWithFeatures.select("NoOfReps", "features").withColumnRenamed("NoOfReps", "label");
    modelInputData.show();

    LinearRegression linearRegression = new LinearRegression();
    LinearRegressionModel model = linearRegression.fit(modelInputData);
    System.out.println(
        "The model has intercept "
            + model.intercept()
            + " and coeffcients "
            + model.coefficients());

    model.transform(modelInputData).show();
  }
}
