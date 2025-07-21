package com.radja.random_forest;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorSlicer;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.IntStream;

public class Main {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("UNSW-NB15 Random Forest Classifier")
                .master("local[*]")
                .config("spark.local.dir", "D:/spark-temp")
                .config("spark.mongodb.input.uri", "mongodb://localhost:27017/NUSW_NB15.network_sessions")
                .config("spark.mongodb.output.uri", "mongodb://localhost:27017/NUSW_NB15.model_outputs")
                .getOrCreate();

        Dataset<Row> rawData = spark.read()
                .format("mongo")
                .load();

        String[] featureCols = new String[]{
                "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl",
                "sloss", "dloss", "Sload", "Dload", "Spkts", "Dpkts",
                "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
                "trans_depth", "res_bdy_len", "Sjit", "Djit",
                "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
                "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
                "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
                "ct_dst_ltm", "ct_src_ ltm", "ct_src_dport_ltm",
                "ct_dst_sport_ltm", "ct_dst_src_ltm"
        };

        Dataset<Row> data = rawData;
        for (String col : featureCols) {
            data = data.withColumn(col, data.col(col).cast("double"));
        }
        data = data.withColumn("Label", data.col("Label").cast("double"));
        data = data.na().drop();

        for (String col : featureCols) {
            data = data.withColumn(col, functions.round(data.col(col), 3));
        }

        long originalCount = data.count();
        Dataset<Row> dedupedData = data.dropDuplicates();
        long dedupCount = dedupedData.count();
        System.out.println("Original count: " + originalCount);
        System.out.println("After deduplication: " + dedupCount);

        Dataset<Row> classCounts = dedupedData.groupBy("Label").count();
        long countClass0 = classCounts.filter("Label = 0").first().getLong(1);
        long countClass1 = classCounts.filter("Label = 1").first().getLong(1);
        double total = countClass0 + countClass1;
        double weight0 = total / (2.0 * countClass0);
        double weight1 = total / (2.0 * countClass1);

        Dataset<Row> weightedData = dedupedData.withColumn("weight",
                functions.when(functions.col("Label").equalTo(0), weight0)
                        .otherwise(weight1));

        VectorAssembler assemblerAll = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        RandomForestClassifier rfAll = new RandomForestClassifier()
                .setLabelCol("Label")
                .setFeaturesCol("features")
                .setWeightCol("weight")
                .setPredictionCol("prediction")
                .setProbabilityCol("probability")
                .setRawPredictionCol("rawPrediction")
                .setNumTrees(100);
//first pipeline with all features passed
        Pipeline pipelineAll = new Pipeline().setStages(new org.apache.spark.ml.PipelineStage[]{assemblerAll, rfAll});

        Dataset<Row>[] splits = weightedData.randomSplit(new double[]{0.8, 0.2}, 12345);
        Dataset<Row> trainAll = splits[0];
        Dataset<Row> testAll = splits[1];

        PipelineModel modelAll = pipelineAll.fit(trainAll);
        RandomForestClassificationModel rfModel = (RandomForestClassificationModel) modelAll.stages()[1];
        Vector importances = rfModel.featureImportances();

        int topK = 20;
        Integer[] indices = IntStream.range(0, importances.size()).boxed().toArray(Integer[]::new);
        Arrays.sort(indices, Comparator.comparingDouble(i -> -importances.apply(i)));

        int[] topFeatureIndices = Arrays.stream(indices).limit(topK).mapToInt(Integer::intValue).toArray();

        System.out.println("Top " + topK + " features:");
        for (int i = 0; i < topK; i++) {
            int idx = topFeatureIndices[i];
            System.out.printf("Feature: %s, Importance: %.5f\n", featureCols[idx], importances.apply(idx));
        }

        VectorSlicer slicer = new VectorSlicer()
                .setInputCol("features")
                .setOutputCol("topFeatures")
                .setIndices(topFeatureIndices);

        RandomForestClassifier rfTop = new RandomForestClassifier()
                .setLabelCol("Label")
                .setFeaturesCol("topFeatures")
                .setWeightCol("weight")
                .setPredictionCol("prediction")
                .setProbabilityCol("probability")
                .setRawPredictionCol("rawPrediction")
                .setNumTrees(100);
//second pipeline with only the top 20 features passed as argument
        Pipeline pipelineTop = new Pipeline().setStages(new org.apache.spark.ml.PipelineStage[]{assemblerAll, slicer, rfTop});
        PipelineModel modelTop = pipelineTop.fit(trainAll);
        Dataset<Row> predictions = modelTop.transform(testAll);

        MulticlassClassificationEvaluator f1Eval = new MulticlassClassificationEvaluator()
                .setLabelCol("Label").setPredictionCol("prediction").setMetricName("f1");
        MulticlassClassificationEvaluator accEval = new MulticlassClassificationEvaluator()
                .setLabelCol("Label").setPredictionCol("prediction").setMetricName("accuracy");
        MulticlassClassificationEvaluator precEval = new MulticlassClassificationEvaluator()
                .setLabelCol("Label").setPredictionCol("prediction").setMetricName("weightedPrecision");
        MulticlassClassificationEvaluator recEval = new MulticlassClassificationEvaluator()
                .setLabelCol("Label").setPredictionCol("prediction").setMetricName("weightedRecall");

        System.out.println("F1 Score     = " + f1Eval.evaluate(predictions));
        System.out.println("Accuracy     = " + accEval.evaluate(predictions));
        System.out.println("Precision    = " + precEval.evaluate(predictions));
        System.out.println("Recall       = " + recEval.evaluate(predictions));

        // === Per-Class Metrics ===
        long TP = predictions.filter("Label = 1 AND prediction = 1").count();
        long TN = predictions.filter("Label = 0 AND prediction = 0").count();
        long FP = predictions.filter("Label = 0 AND prediction = 1").count();
        long FN = predictions.filter("Label = 1 AND prediction = 0").count();

        double precision_class1 = TP / (double)(TP + FP + 1e-10); // avoid div zero
        double recall_class1 = TP / (double)(TP + FN + 1e-10);
        double f1_class1 = 2 * precision_class1 * recall_class1 / (precision_class1 + recall_class1 + 1e-10);

        double precision_class0 = TN / (double)(TN + FN + 1e-10);
        double recall_class0 = TN / (double)(TN + FP + 1e-10);
        double f1_class0 = 2 * precision_class0 * recall_class0 / (precision_class0 + recall_class0 + 1e-10);

        System.out.printf("Class 0: Precision = %.4f, Recall = %.4f, F1 = %.4f%n", precision_class0, recall_class0, f1_class0);
        System.out.printf("Class 1: Precision = %.4f, Recall = %.4f, F1 = %.4f%n", precision_class1, recall_class1, f1_class1);

        // === Prepare Output ===
        spark.udf().register("vectorToArray", (UDF1<Vector, double[]>) Vector::toArray,
                DataTypes.createArrayType(DataTypes.DoubleType));

        Dataset<Row> output = predictions
                .withColumn("features_array", functions.callUDF("vectorToArray", functions.col("topFeatures")))
                .withColumn("probability_array", functions.callUDF("vectorToArray", functions.col("probability")))
                .withColumn("rawPrediction_array", functions.callUDF("vectorToArray", functions.col("rawPrediction")))
                .select(
                        functions.col("prediction"),
                        functions.col("Label"),
                        functions.col("features_array").alias("features"),
                        functions.col("probability_array").alias("probability"),
                        functions.col("rawPrediction_array").alias("rawPrediction")
                );

        // === Save to MongoDB ===
        output.write()
                .format("mongo")
                .mode("overwrite")
                .option("collection", "model_output" +
                        "s")
                .save();

        // === Save as CSV to single file ===
        Dataset<Row> outputForCsv = output
                .withColumn("features_str", functions.concat_ws(",", functions.col("features")))
                .withColumn("probability_str", functions.concat_ws(",", functions.col("probability")))
                .withColumn("rawPrediction_str", functions.concat_ws(",", functions.col("rawPrediction")))
                .select(
                        functions.col("prediction"),
                        functions.col("Label"),
                        functions.col("features_str"),
                        functions.col("probability_str"),
                        functions.col("rawPrediction_str")
                );

        String tempDir = "C:/Users/DELL/Documents/rf_predictions_temp";
        String finalCsvPath = "C:/Users/DELL/Documents/rf_predictions.csv";

        outputForCsv.coalesce(1)
                .write()
                .format("csv")
                .option("header", "true")
                .mode("overwrite")
                .save(tempDir);

        try {
            java.nio.file.Path tempPath = java.nio.file.Paths.get(tempDir);
            java.nio.file.Path finalPath = java.nio.file.Paths.get(finalCsvPath);

            // Find the part-*.csv file
            java.nio.file.DirectoryStream<java.nio.file.Path> dirStream = java.nio.file.Files.newDirectoryStream(tempPath);
            boolean fileMoved = false;
            for (java.nio.file.Path file : dirStream) {
                String name = file.getFileName().toString();
                if (name.startsWith("part-") && name.endsWith(".csv")) {
                    java.nio.file.Files.move(file, finalPath, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                    System.out.println("Moved file: " + file + " to " + finalPath);
                    fileMoved = true;
                    break;
                }
            }
            dirStream.close();

            if (!fileMoved) {
                System.err.println("No part-*.csv file found in " + tempDir);
            }

            // Delete all files inside temp folder (including _SUCCESS and CRC files)
            try (java.nio.file.DirectoryStream<java.nio.file.Path> cleanupStream = java.nio.file.Files.newDirectoryStream(tempPath)) {
                for (java.nio.file.Path file : cleanupStream) {
                    java.nio.file.Files.deleteIfExists(file);
                }
            }

            // Now delete the empty temp folder itself
            java.nio.file.Files.deleteIfExists(tempPath);

            System.out.println("Cleaned up temp folder: " + tempDir);

        } catch (Exception e) {
            System.err.println("Error renaming/cleaning CSV file: " + e.getMessage());
            e.printStackTrace();
        }
    }}

