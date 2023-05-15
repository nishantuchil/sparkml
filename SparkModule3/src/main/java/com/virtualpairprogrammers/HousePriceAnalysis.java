package com.virtualpairprogrammers;

import static org.apache.spark.sql.functions.col;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class HousePriceAnalysis {

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

    csvData =
        csvData.withColumn("sqft_above_percentage", col("sqft_above").divide(col("sqft_living")));

    csvData = csvData.withColumnRenamed("price", "label");

    Dataset<Row>[] dataSplits = csvData.randomSplit(new double[] {0.8, 0.2});
    Dataset<Row> trainingAndTestData = dataSplits[0];
    Dataset<Row> holdOutData = dataSplits[1];

    StringIndexer stringIndexer = new StringIndexer();
    stringIndexer.setInputCols(new String[] {"condition", "grade", "zipcode"});
    stringIndexer.setOutputCols(new String[] {"conditionIndex", "gradeIndex", "zipcodeIndex"});
    // csvData = stringIndexer.fit(csvData).transform(csvData);

    OneHotEncoder oneHotEncoder = new OneHotEncoder();
    oneHotEncoder.setInputCols(new String[] {"conditionIndex", "gradeIndex", "zipcodeIndex"});
    oneHotEncoder.setOutputCols(new String[] {"conditionVector", "gradeVector", "zipcodeVector"});
    // csvData = oneHotEncoder.fit(csvData).transform(csvData);

    // csvData.show();

    VectorAssembler vectorAssembler = new VectorAssembler();
    vectorAssembler.setInputCols(
        new String[] {
          "bedrooms",
          "bathrooms",
          "sqft_living",
          "sqft_above_percentage",
          "floors",
          "conditionVector",
          "gradeVector",
          "zipcodeVector",
          "waterfront"
        });
    vectorAssembler.setOutputCol("features");

    // Dataset<Row> csvDatawithFeatures = vectorAssembler.transform(csvData);

    // Dataset<Row> modelInputData =
    //    csvDatawithFeatures.select("price", "features").withColumnRenamed("price", "label");

    // csvDatawithFeatures.printSchema();
    // csvDatawithFeatures.show(false);

    // modelInputData.show();

    // Dataset<Row>[] dataSplit = modelInputData.randomSplit(new double[] {0.8, 0.2});

    // Dataset<Row> trainingAndTestData = dataSplit[0];
    // Dataset<Row> holdData = dataSplit[1];

    LinearRegression linearRegression = new LinearRegression();

    ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
    ParamMap[] paramMap =
        paramGridBuilder
            .addGrid(linearRegression.regParam(), new double[] {0.01, 0.1, 0.5})
            .addGrid(linearRegression.elasticNetParam(), new double[] {0, 0.5, 1})
            .build();

    TrainValidationSplit trainValidationSplit =
        new TrainValidationSplit()
            .setEstimator(linearRegression)
            .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
            .setEstimatorParamMaps(paramMap)
            .setTrainRatio(0.8);

    // TrainValidationSplitModel model = trainValidationSplit.fit(trainingAndTestData);
    // LinearRegressionModel lrModel = (LinearRegressionModel) model.bestModel();

    Pipeline pipeline = new Pipeline();
    pipeline.setStages(
        new PipelineStage[] {stringIndexer, oneHotEncoder, vectorAssembler, trainValidationSplit});
    PipelineModel pipelineModel = pipeline.fit(trainingAndTestData);
    TrainValidationSplitModel model = (TrainValidationSplitModel) pipelineModel.stages()[3];

    LinearRegressionModel lrModel = (LinearRegressionModel) model.bestModel();

    System.out.println(
        "RMSE is " + lrModel.summary().rootMeanSquaredError() + " R2 is " + lrModel.summary().r2());

    holdOutData = pipelineModel.transform(holdOutData).drop("prediction");

    // lrModel.transform(holdOutData).show();

    System.out.println(
        "RMSE is "
            + lrModel.evaluate(holdOutData).rootMeanSquaredError()
            + " R2 is "
            + lrModel.evaluate(holdOutData).r2());

    System.out.println(
        "coefficients " + lrModel.coefficients() + " intercept " + lrModel.intercept());
    System.out.println(
        "reg param "
            + lrModel.getRegParam()
            + " elastic net param "
            + lrModel.getElasticNetParam());

    spark.close();
  }
}
