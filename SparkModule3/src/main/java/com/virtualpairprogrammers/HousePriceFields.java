package com.virtualpairprogrammers;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class HousePriceFields {

  public static void main(String args[]) {

    LoggerContext logContext = (LoggerContext) LogManager.getContext(false);
    Configuration logConfiguration = logContext.getConfiguration();
    LoggerConfig loggerConfig = logConfiguration.getLoggerConfig("org.apache");
    loggerConfig.setLevel(Level.WARN);
    logContext.updateLoggers();

    SparkSession spark =
        SparkSession.builder()
            .appName("HousePriceAnalysis")
            .config("spark.sql.warehouse.dir", "file:///c:/tmp/")
            .master("local[*]")
            .getOrCreate();

    Dataset<Row> csvData =
        spark
            .read()
            .option("header", true)
            .option("inferSchema", true)
            .csv("src/main/resources/kc_house_data.csv");

    csvData.describe().show();

    csvData =
        csvData.drop(
            "id",
            "date",
            "waterfront",
            "view",
            "condition",
            "grade",
            "yr_renovated",
            "zipcode",
            "lat",
            "lang");

    for (String col : csvData.columns()) {
      System.out.println(
          "Correlation betweeen price and " + col + " is " + csvData.stat().corr("price", col));
    }

    csvData = csvData.drop("sqft_lot", "sqft_lot15", "yr_built", "sqft_living15");

    for (String outCol : csvData.columns()) {
      for (String innerCol : csvData.columns()) {
        System.out.println(
            "Correlation betweeen "
                + outCol
                + " and "
                + innerCol
                + " is "
                + csvData.stat().corr(outCol, innerCol));
      }
    }
    spark.close();
  }
}
