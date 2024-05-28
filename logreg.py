from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder.getOrCreate()

train = spark.read.csv('s3://cs4296assignment/mnist_train.csv',header=True, inferSchema=True)
#train = spark.read.csv("c:/Users/Peiran/OneDrive - National University of Singapore/Y3S2 (Exchange)/CS4296/assignment3/mnist_train.csv", header=True, inferSchema=True)
test = spark.read.csv('s3://cs4296assignment/mnist_test.csv')
#test = spark.read.csv("c:/Users/Peiran/OneDrive - National University of Singapore/Y3S2 (Exchange)/CS4296/assignment3/mnist_test.csv", header=True, inferSchema=True)

input_columns = [f'pixel{i}' for i in range(784)]
assembler = VectorAssembler(inputCols=input_columns, outputCol="features")
train_transformed = assembler.transform(train)
test_transformed = assembler.transform(test)

training_df = train_transformed.select(['features','label'])

log_reg=LogisticRegression(labelCol='label').fit(training_df)

train_results=log_reg.evaluate(training_df).predictions
#train_results.show()
res = "Training Accuracy: " + str(train_results[(train_results.label == train_results.prediction)].count()/train_results.count())

results=log_reg.evaluate(test_transformed).predictions
#results.show()
res += ("\n Test Accuracy: " + str(results[(results.label==results.prediction)].count()/results.count()))
#print(res)
sc = spark.sparkContext
result_rdd = sc.parallelize([res])
result_rdd.saveAsTextFile("s3://cs4296assignment/result.txt")
