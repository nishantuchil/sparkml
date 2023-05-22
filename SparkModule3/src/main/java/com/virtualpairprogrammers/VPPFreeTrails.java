package com.virtualpairprogrammers;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.lit;
import static org.apache.spark.sql.functions.when;

import java.util.Arrays;
import java.util.List;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

public class VPPFreeTrails {

  public static UDF1<String, String> countryGrouping =
      new UDF1<String, String>() {
        @Override
        public String call(String country) throws Exception {
          List<String> topCountries = Arrays.asList(new String[] {"GB", "US", "IN", "UNKNOWN"});
          List<String> europeanCountries =
              Arrays.asList(
                  new String[] {
                    "BE", "BG", "CZ", "DK", "DE", "EE", "IE", "EL", "ES", "FR", "HR", "IT", "CY",
                    "LV", "LT", "LU", "HU", "MT", "NL", "AT", "PL", "PT", "RO", "SI", "SK", "FI",
                    "SE", "CH", "IS", "NO", "LI", "EU"
                  });

          if (topCountries.contains(country)) return country;
          if (europeanCountries.contains(country)) return "EUROPE";
          else return "OTHER";
        }
      };

  public static void main(String args[]) {

    System.setProperty(
        "hadoop.home.dir", "D:\\udemy\\ApacheSpark\\Practicals\\winutils-extra\\hadoop");

    LoggerContext loggerContext = (LoggerContext) LogManager.getContext(false);
    Configuration loggerConfiguration = loggerContext.getConfiguration();
    loggerConfiguration.getLoggerConfig("org.apache").setLevel(Level.WARN);
    loggerContext.updateLoggers();

    SparkSession sparkSession =
        SparkSession.builder()
            .appName("VPPFreeTrails")
            .master("local[*]")
            .config("spark.sql.warehouse.dir", "file:///c:/tmp/")
            .getOrCreate();

    Dataset<Row> csvData =
        sparkSession
            .read()
            .option("header", true)
            .option("inferSchema", true)
            .csv("src/main/resources/vppFreeTrials.csv");

    sparkSession.udf().register("countryGrouping", countryGrouping, DataTypes.StringType);

    csvData =
        csvData
            .withColumn("country", callUDF("countryGrouping", col("country")))
            .withColumn("label", when(col("payments_made").geq(1), lit(1)).otherwise(lit(0)));

    StringIndexer countryIndexer = new StringIndexer();
    csvData =
        countryIndexer
            .setInputCol("country")
            .setOutputCol("countryIndex")
            .fit(csvData)
            .transform(csvData);

    new IndexToString()
        .setInputCol("countryIndex")
        .setOutputCol("value")
        .transform(csvData.select("countryIndex").distinct())
        .show();

    VectorAssembler vectorAssembler = new VectorAssembler();
    vectorAssembler.setInputCols(
        new String[] {"countryIndex", "rebill_period", "chapter_access_count", "seconds_watched"});
    vectorAssembler.setOutputCol("features");

    Dataset<Row> inputData = vectorAssembler.transform(csvData).select("label", "features");
    inputData.show();

    Dataset<Row>[] trainingAndHoldoutData = inputData.randomSplit(new double[] {0.9, 0.1});
    Dataset<Row> trainingData = trainingAndHoldoutData[0];
    Dataset<Row> holdoutData = trainingAndHoldoutData[1];

    DecisionTreeClassifier dtClassifer = new DecisionTreeClassifier();
    dtClassifer.setMaxDepth(3);

    DecisionTreeClassificationModel model = dtClassifer.fit(trainingData);

    Dataset<Row> predictions = model.transform(holdoutData);
    predictions.show();

    System.out.println(model.toDebugString());

    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
    evaluator.setMetricName("accuracy");
    System.out.println(evaluator.evaluate(predictions));

    RandomForestClassifier rfClassifier = new RandomForestClassifier();
    rfClassifier.setMaxDepth(3);
    RandomForestClassificationModel rfModel = rfClassifier.fit(trainingData);
    Dataset<Row> predictions2 = rfModel.transform(holdoutData);
    predictions2.show();
    System.out.println(rfModel.toDebugString());
    System.out.println(evaluator.evaluate(predictions2));
  }
}
