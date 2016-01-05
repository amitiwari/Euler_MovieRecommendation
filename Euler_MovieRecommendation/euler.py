from pyspark import SparkContext
from pyspark.sql import SQLContext 
from pyspark import SparkConf
import sys, operator
import re, string, random
from itertools import repeat


inputs = int(sys.argv[1])
output = sys.argv[2]

conf = SparkConf().setAppName('euler')
sc = SparkContext()

part = 100

def add_tuples(a, b):
    return tuple(sum(p) for p in zip(a,b))     
    
def gen_euler(num):
    rand = random.Random()
    count=0
    for loop in xrange(num):
        sum=0.0
        while sum<1:
            sum = sum + rand.random()
            count += 1
            
    return (num,count)

#mydata = list(repeat(part, inputs/part))

parallel_input = sc.parallelize([inputs/part]*part, part)

eluerno = parallel_input.map(gen_euler).reduce(add_tuples)

save_euler = sc.parallelize([(eluerno[0], eluerno[1])], numSlices=1)
save_euler.saveAsTextFile(output)