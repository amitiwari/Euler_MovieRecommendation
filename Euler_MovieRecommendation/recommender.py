from pyspark import SparkContext
from pyspark.sql import SQLContext 
from pyspark import SparkConf
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import sys, operator
import re, string, unicodedata
from pyspark.sql.functions import levenshtein
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

inputs = sys.argv[1]
userrating = sys.argv[2]
output = sys.argv[3]

conf = SparkConf().setAppName('recommend')
sc = SparkContext()
sqlContext = SQLContext(sc)

movies_rdd = sc.textFile(inputs + '/movies.dat')
ratings_rdd = sc.textFile(inputs + '/ratings.dat')
userrating_rdd = sc.textFile(userrating)

movies_split = movies_rdd.map(lambda line: line.split("::")) \
    .map(lambda (movieid,movietitle,genre):(movieid,movietitle)).map(lambda (movieid,movietitle): (int(movieid),(unicodedata.normalize('NFD', movietitle))))
 
rating_split = ratings_rdd.map(lambda line: line.split("::")) \
    .map(lambda (userid,movieid,rating,timestamp):(int(userid),int(movieid),rating))
    
    
userrating_split = userrating_rdd.map(lambda line: line.split(" ",1)) \
    .map(lambda (userrating,usertitle): (userrating,(unicodedata.normalize('NFD', usertitle)), 0))
    
userschema = StructType([
    StructField('userrating', StringType(),  False),
    StructField('usermovietitle', StringType(), False),
    StructField('userid', IntegerType(),  False)
])

movieschema = StructType([
    StructField('movieid', IntegerType(), False),
    StructField('tweetmovietitle', StringType(), False)
])

ratingschema = StructType([
    StructField('userid', IntegerType(), False),
    StructField('movieid', IntegerType(), False),
    StructField('rating', StringType(), False)
])

userrating_sql = sqlContext.createDataFrame(userrating_split, userschema)
movies_sql = sqlContext.createDataFrame(movies_split, movieschema).cache()
rating_sql = sqlContext.createDataFrame(rating_split, ratingschema)

movie_prep = movies_sql.select('movieid')

movies_join_usersrating = userrating_sql.join(movies_sql)
rating_join_movies = rating_sql.join(movie_prep,['movieid'])


distmovies = movies_join_usersrating.select('movieid', 'tweetmovietitle', 'userrating', 'usermovietitle', 'userid',levenshtein('usermovietitle', 'tweetmovietitle').alias('min-dist')).cache()

mindistmovies =  distmovies.groupBy('usermovietitle').min('min-dist').withColumnRenamed('min(min-dist)', 'min-dist')

user_joined = mindistmovies.join(distmovies, ['usermovietitle', 'min-dist']).select('userid', 'movieid', 'userrating')

train_data = user_joined.unionAll(rating_join_movies).cache()
rank = 10
numIterations = 10
model = ALS.train(train_data, rank, numIterations)

movies = model.recommendProducts(0, 10)

movies_rdd = sc.parallelize(movies, 1)

moviespredict = sqlContext.createDataFrame(movies_rdd, ratingschema)
movies_final = moviespredict.join(movies_sql, ['movieid']).select('tweetmovietitle')
movies_final.rdd.map(lambda row: row[0]).repartition(1).saveAsTextFile(output)