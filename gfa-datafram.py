# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 20:51:43 2022

@author: HP
"""

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

df = pd.read_csv('C:/Users/HP/Desktop/ir4.csv')

# Create the Session
spark = SparkSession.builder \
    .master("local") \
    .appName("PySpark Tutorial") \
    .getOrCreate()

sc = spark.sparkContext

df1 = spark.createDataFrame(df)
vecAssembler = VectorAssembler(inputCols=["lat", "lng"], outputCol="features")
new_df = vecAssembler.transform(df1)
new_df.show()

kmeans = KMeans(k=2, seed=1)
model = kmeans.fit(new_df.select('features'))
transformed = model.transform(new_df)
transformed.show()

# rdd = sc.textFile('C:/Users/HP/Desktop/ir4.csv')
# rdd.take(1)

# rdd1 = rdd.map(lambda line: line.split(","))
# rdd1.top(5)
# stock_1 = spark.read.csv('C:/Users/HP/Desktop/-PA.TO.csv',\
# inferSchema=True, header=True)
# stock_1.show(5)

