package com.radja.random_forest;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;

public class Main {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("UNSW-NB15 Random Forest Classifier")
                .master("local[*]")
                .config("spark.mongodb.input.uri", "mongodb://localhost:27017/NUSW_NB15.network_sessions")
                .config("spark.mongodb.output.uri", "mongodb://localhost:27017/NUSW_NB15.model_output")
                .getOrCreate();

        // Read data from MongoDB
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

        // Cast features to double
        for (String col : featureCols) {
            rawData = rawData.withColumn(col, rawData.col(col).cast("double"));
        }

        // Cast label to double
        Dataset<Row> data = rawData.withColumn("Label", rawData.col("Label").cast("double"));

        // Drop nulls
        data = data.na().drop();

        // Assemble features vector
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        // Random Forest classifier
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("Label")
                .setFeaturesCol("features")
                .setPredictionCol("prediction")
                .setProbabilityCol("probability")
                .setNumTrees(100);

        // Pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{assembler, rf});

        // Train/test split
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.8, 0.2}, 12345);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train model
        PipelineModel model = pipeline.fit(trainingData);

        // Predict on test
        Dataset<Row> predictions = model.transform(testData);

        // Evaluate F1
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("Label")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1Score = evaluator.evaluate(predictions);
        System.out.println("F1 Score on test data = " + f1Score);

        // Evaluate accuracy
        MulticlassClassificationEvaluator accuracyEval = new MulticlassClassificationEvaluator()
                .setLabelCol("Label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        System.out.println("Accuracy = " + accuracyEval.evaluate(predictions));

        // Register UDF to convert Vector to double[]
        UDF1<Vector, double[]> vectorToArray = new UDF1<Vector, double[]>() {
            @Override
            public double[] call(Vector v) {
                return v.toArray();
            }
        };
        spark.udf().register("vectorToArray", vectorToArray, DataTypes.createArrayType(DataTypes.DoubleType));

        // Convert "probability" and "features" columns from Vector to array before saving
        Dataset<Row> output = predictions
                .withColumn("probability", functions.callUDF("vectorToArray", predictions.col("probability")))
                .withColumn("features", functions.callUDF("vectorToArray", predictions.col("features")))
                .select("prediction", "probability", "Label", "features");

        // Save to MongoDB
        output.write()
                .format("mongo")
                .mode("overwrite")
                .option("collection", "model_output")
                .save();

        spark.stop();
    }
}





