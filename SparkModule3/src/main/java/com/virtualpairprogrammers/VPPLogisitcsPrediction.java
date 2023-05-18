package com.virtualpairprogrammers;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.when;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionSummary;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class VPPLogisitcsPrediction {

  public static void main(String args[]) {

    LoggerContext loggerContext = (LoggerContext) LogManager.getContext(false);
    Configuration loggerConfiguration = loggerContext.getConfiguration();
    loggerConfiguration.getLoggerConfig("org.apache").setLevel(Level.WARN);
    loggerContext.updateLoggers();

    SparkSession sparkSession =
        SparkSession.builder()
            .appName("CustomerSubscriptionPrediction")
            .master("local[*]")
            .config("spark.sql.warehouse.dir", "file:///c:/tmp/")
            .getOrCreate();

    Dataset<Row> initialDataset =
        sparkSession
            .read()
            .option("header", true)
            .option("inferSchema", true)
            .csv(
                "D:\\udemy\\ApacheSpark\\Practicals\\Code\\Starting workspaces\\Chapter 08\\vppChapterViews\\part-r-00000-d55d9fed-7427-4d23-aa42-495275510f78.csv",
                "D:\\udemy\\ApacheSpark\\Practicals\\Code\\Starting workspaces\\Chapter 08\\vppChapterViews\\part-r-00001-d55d9fed-7427-4d23-aa42-495275510f78.csv",
                "D:\\udemy\\ApacheSpark\\Practicals\\Code\\Starting workspaces\\Chapter 08\\vppChapterViews\\part-r-00002-d55d9fed-7427-4d23-aa42-495275510f78.csv",
                "D:\\udemy\\ApacheSpark\\Practicals\\Code\\Starting workspaces\\Chapter 08\\vppChapterViews\\part-r-00003-d55d9fed-7427-4d23-aa42-495275510f78.csv");

    Dataset<Row> filterDataSet =
        initialDataset
            .filter(col("is_cancelled").equalTo(false))
            .withColumnRenamed("next_month_views", "label")
            .drop("observation_date", "is_cancelled");

    filterDataSet =
        filterDataSet
            .withColumn("firstSub", when(col("firstSub").isNull(), 0).otherwise(col("firstSub")))
            .withColumn(
                "all_time_views",
                when(col("all_time_views").isNull(), 0).otherwise(col("all_time_views")))
            .withColumn(
                "last_month_views",
                when(col("last_month_views").isNull(), 0).otherwise(col("last_month_views")));
    filterDataSet =
        filterDataSet.withColumn("label", when(col("label").$greater(0), 0).otherwise(1));

    Dataset<Row> dataSplits[] = filterDataSet.randomSplit(new double[] {0.9, 0.1});
    Dataset<Row> testData = dataSplits[0];
    Dataset<Row> holdOutData = dataSplits[1];

    StringIndexer stringIndexer = new StringIndexer();
    stringIndexer.setInputCols(
        new String[] {"payment_method_type", "country", "rebill_period_in_months"});
    stringIndexer.setOutputCols(new String[] {"payment_index", "country_index", "rebill_index"});

    OneHotEncoder oneHotEncoder = new OneHotEncoder();
    oneHotEncoder.setInputCols(new String[] {"payment_index", "country_index", "rebill_index"});
    oneHotEncoder.setOutputCols(
        new String[] {"payment_encoded", "country_encoded", "rebill_encoded"});

    VectorAssembler vectorAssembler = new VectorAssembler();
    vectorAssembler.setInputCols(
        new String[] {
          "payment_encoded",
          "country_encoded",
          "rebill_encoded",
          "firstSub",
          "age",
          "all_time_views",
          "last_month_views",
        });
    vectorAssembler.setOutputCol("features");

    LogisticRegression logisticRegression = new LogisticRegression();

    ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
    ParamMap paramMap[] =
        paramGridBuilder
            .addGrid(logisticRegression.regParam(), new double[] {0.01, 0.1, 0.3, 0.5, 0.7, 1})
            .addGrid(logisticRegression.elasticNetParam(), new double[] {0, 0.5, 1})
            .build();

    TrainValidationSplit trainValidationSplit =
        new TrainValidationSplit()
            .setEstimator(logisticRegression)
            .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
            .setEstimatorParamMaps(paramMap)
            .setTrainRatio(0.9);

    Pipeline pipeline = new Pipeline();
    pipeline.setStages(
        new PipelineStage[] {stringIndexer, oneHotEncoder, vectorAssembler, trainValidationSplit});
    PipelineModel pipelineModel = pipeline.fit(testData);
    TrainValidationSplitModel model = (TrainValidationSplitModel) pipelineModel.stages()[3];
    LogisticRegressionModel lrm = (LogisticRegressionModel) model.bestModel();

    System.out.println("Accuracy for test data = " + lrm.summary().accuracy());

    Dataset<Row> predictedData = pipelineModel.transform(holdOutData);
    predictedData = predictedData.drop(col("prediction"));
    predictedData = predictedData.drop(col("rawPrediction"));
    predictedData = predictedData.drop(col("probability"));

    LogisticRegressionSummary evaluationResult = lrm.evaluate(predictedData);

    double truePositives = evaluationResult.truePositiveRateByLabel()[1];
    double falsePositives = evaluationResult.falsePositiveRateByLabel()[0];

    System.out.println(
        "for hold out data the likelihood of a positive being correct is "
            + truePositives / (truePositives + falsePositives));
    System.out.println("accuracy for model is " + evaluationResult.accuracy());

    lrm.transform(predictedData).groupBy("label", "prediction").count().show();

    sparkSession.close();
  }
}
