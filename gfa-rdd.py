
import numpy as np
from pyspark.sql import SparkSession
from pyspark.mllib.clustering import KMeans


def parseVector(line):
    return np.array([float(x) for x in line.split(",")])


if __name__ == "__main__":
    spark = SparkSession.builder \
        .master("local") \
        .appName("PySpark Tutorial") \
        .getOrCreate()
    sc = spark.sparkContext

    lines = sc.textFile('C:/Users/HP/Desktop/ir5.csv')
    header = lines.first()  # extract header
    lines = lines.filter(lambda x: x != header) # remove header
    data = lines.map(parseVector)

    k = 3
    model = KMeans.train(data, k)
    print("Final centers: " + str(model.clusterCenters))
    print("Total Cost: " + str(model.computeCost(data)))
    sc.stop()