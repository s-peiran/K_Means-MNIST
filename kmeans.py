from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.sql.window import Window
from pyspark.sql.functions import col, rank, desc

spark = SparkSession.builder.getOrCreate()

#train = spark.read.csv('s3://cs4296assignment/mnist_train.csv',header=True, inferSchema=True)
train = spark.read.csv("c:/Users/Peiran/OneDrive - National University of Singapore/Y3S2 (Exchange)/CS4296/assignment3/mnist_train.csv", header=True, inferSchema=True)
#test = spark.read.csv('s3://cs4296assignment/mnist_test.csv')
test = spark.read.csv("c:/Users/Peiran/OneDrive - National University of Singapore/Y3S2 (Exchange)/CS4296/assignment3/mnist_test.csv", header=True, inferSchema=True)

input_columns = [f'pixel{i}' for i in range(784)]
assembler = VectorAssembler(inputCols=input_columns, outputCol="features")
# Transform the DataFrame to create the 'features' vector column    
df_transformed = assembler.transform(train) 
#df_transformed['features']

test_transformed = assembler.transform(test)

kmeans = KMeans(featuresCol="features",k=100)
model = kmeans.fit(df_transformed)
predictions = model.transform(df_transformed)
test_predictions = model.transform(test_transformed)
#predictions.groupBy("label",'prediction').count().orderBy("label").show(n=1000, truncate=False)

#train.groupBy("label").count().orderBy("label").show()

cluster_label_mode = predictions.groupBy("prediction", "label").count()
window = Window.partitionBy("prediction").orderBy(col("count").desc())  

# Using the Window function to find the most frequent label per cluster
cluster_label_mode = cluster_label_mode.withColumn("rank", rank().over(window)) \
                                       .filter(col("rank") == 1) \
                                       .drop("rank", "count")\
                                       .withColumnRenamed("label","most_frequent_label")

# Step 2: Join the most common label back to the original predictions to assign each point a 'predicted_label'
predictions = predictions.join(cluster_label_mode, "prediction", "left_outer")

# Step 3: Calculate accuracy
correct_predictions = predictions.where(predictions["label"] == predictions["most_frequent_label"]).count()
total_predictions = predictions.count()
accuracy = correct_predictions / total_predictions
res ="Training Accuracy: " + str(accuracy)

test_predictions_results = test_predictions.join(cluster_label_mode, "prediction", "left_outer")
correct_test_predictions = test_predictions_results.where(test_predictions_results["label"] == test_predictions_results["most_frequent_label"]).count()
total_test_predictions = test_predictions_results.count()
test_accuracy = correct_test_predictions / total_test_predictions
res += ("\n"+"Test Accuracy: " + str(test_accuracy))
print(res)
#sc = spark.sparkContext
#result_rdd = sc.parallelize([res])
#result_rdd.saveAsTextFile("s3://cs4296assignment/assignment3/result.txt")

#print("Accuracy:", accuracy)
