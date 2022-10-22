
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('C:/Users/HP/Desktop/ir2.csv')
X_weighted =df.loc[:,['Latitude','Longitude','kgg']]

# Create the Session
spark = SparkSession.builder \
    .master("local") \
    .appName("PySpark Tutorial") \
    .getOrCreate()

sc = spark.sparkContext

df1 = spark.createDataFrame(X_weighted)
vecAssembler = VectorAssembler(inputCols=["Latitude", "Longitude"], outputCol="features")
new_df = vecAssembler.transform(df1)
new_df.show()

kmeans = KMeans(k=10, seed=1, featuresCol='features', weightCol='kgg', maxIter=200, initSteps=8)
model = kmeans.fit(new_df)
prediction = model.transform(new_df)
prediction.show()

centers = model.clusterCenters()
print(centers)

summary = model.summary
print(summary.k)
print(summary.clusterSizes)
print(summary.trainingCost)
