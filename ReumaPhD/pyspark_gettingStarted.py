import os

import numpy as np
import pyspark

os.environ['PYSPARK_PYTHON'] = r'C:\ProgramData\Anaconda3\python.exe'
os.environ['HADOOP_HOME'] = r'C:\ProgramData\winutils'


def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    xy2 = x ** 2 + y ** 2
    elevation = np.arctan2(z, np.sqrt(xy2))
    range = np.sqrt(xy2 + z ** 2)

    elevation *= 180. / np.pi
    azimuth *= 180. / np.pi
    if azimuth < 0:
        azimuth += 360.

    return float(elevation), float(azimuth), float(range)


# first program in Spark
# -------------------------

# num_samples = 100000000
# conf = pyspark.SparkConf()
#
#
# def inside(p):
#     x, y = random.random(), random.random()
#     return x*x + y*y < 1
#
#
# sc = pyspark.SparkContext(appName="Pi")
# count = sc.parallelize(range(0, num_samples)).filter(inside).count()
#
# pi = 4 * count / num_samples
# print(pi)
#
# sc.stop()


# second program in pyspark
# --------------------------
sc = pyspark.SparkContext(appName = 'pyspark tutorial')
# temp_c = [10, 3, -5, 25, 1, 9, 29, -10]
# rdd_temp_c = sc.parallelize(temp_c)
# rdd_temp_K = rdd_temp_c.map(lambda x: x + 273.15).collect() # the map transforms the list while the collect pulls the
#                                                             # transformed numbers to the driver
#
# # Use reduce as an action to combine numbers
# rdd_combined = rdd_temp_c.reduce(lambda x, y: x + y)
# print(rdd_combined)


# Read text file in spark
# read input text file to RDD
pts = sc.textFile(r"D:\OwnCloud\Data\PCLs\shmulik_small.txt")
df = pts.toDF(["X", "Y", "Z"])
# split the data where there is a comma
sph_rdd = pts.map(lambda x: x.split(',')).map(lambda y: cart2sph(float(y[0]), float(y[1]), float(y[2])))

# print the list

df.printSchema()
# # # sph = sc.parallelize(pts.filter(SphericalCoordinatesFactory.CartesianToSphericalCoordinates))
#
# print('hello')
#
#
