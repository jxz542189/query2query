# -*- coding: utf-8 -*-
import re
import random
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.sql.functions import rand


today = datetime.now()
profile_date = (today - timedelta(2)).strftime('%Y-%m-%d')
yesterday = today - timedelta(1)
log_date = yesterday.strftime('%Y-%m-%d')
spark = SparkSession.builder.appName("search_jixiaozhan_query").enableHiveSupport().getOrCreate()
spark.conf.set("hive.exec.dynamic.partition", "true")
spark.conf.set("hive.exec.dynamic.partition.mode", "nonstrict")

sc = spark.sparkContext
sc.setLogLevel("WARN")
df_1 = spark.sql("""SELECT name FROM jiayundw_dwd.odoo_product_template_da
WHERE date_id='{log_date}' and active=1 limit 500000
""".format(log_date=log_date))
titles = []

names = df_1.select('name').rdd.collect()
for name in names:
    titles.append(name.name)
titles_broadcast = sc.broadcast(titles)

choice_udf = udf(lambda x: random.choice(titles_broadcast.value), StringType())
word2id_path = "s3://jiayun.spark.data/product_algorithm/sentence_match/word2id.txt"
word2id_df = spark.read.csv(word2id_path, header=None, sep='\t')
word2id_df = word2id_df.withColumnRenamed('_c0', 'word').withColumnRenamed("_c1", "id")
words = word2id_df.rdd.collect()
word2id = {}

for word in words:
    try:
        word2id[re.sub("\xa0", "", str(word.word))] = int(re.sub('\n', '', str(word.id)))
    except:
        pass

word2id_broadcast = sc.broadcast(word2id)
print(len(word2id_broadcast.value))

max_sentence_len = 30


def word_map(sentence):
    words = sentence.lower().split(" ")[:max_sentence_len]
    return [word2id_broadcast.value[w] if w in word2id_broadcast.value else 0 for w in words] + \
           [0] * (max_sentence_len - len(words))


def sentence_len(sentence):
    words = sentence.split(" ")[:max_sentence_len]
    return len(words)


sentence_map_udf = udf(word_map, ArrayType(IntegerType()))
sentence_len_udf = udf(sentence_len, IntegerType())


for day in range(1, 0, -1):
    yesterday = today - timedelta(day)
    log_date = yesterday.strftime('%Y-%m-%d')

    train_path = """s3://jiayun.spark.data/product_algorithm/sentence_match/data/train_{log_date}_tfrecord""".format(log_date=today.strftime('%Y-%m-%d'))
    eval_path = """s3://jiayun.spark.data/product_algorithm/sentence_match/data/eval_{log_date}_tfrecord""".format(log_date=today.strftime('%Y-%m-%d'))


    df = spark.sql("""SELECT u.query, p.name FROM
    (SELECT lower(query) as query, pid as cpid, count(*) as cnt, gender FROM jiayundw_dwd.flow_user_trace_add_da
    WHERE length(query) > 0 AND length(query) < 50
    GROUP BY query, cpid, gender
    ORDER BY cnt DESC, cpid) as u INNER JOIN (SELECT pid, pno, lower(pname) as name, catid1, catid2, catid3
    FROM jiayundw_dm.product_profile_df
    WHERE date_id = '{profile_date}'
    GROUP BY pid, pno, name, catid1, catid2, catid3) as p
    ON u.cpid = p.pid""".format(profile_date=profile_date))
    df = df.repartition(1000)

    # df_2 = df_2.select("query", "name")

    df_3 = spark.sql("""SELECT u.query, p.name FROM
    (SELECT lower(query) as query, pid as cpid, count(*) as cnt, gender FROM jiayundw_dwd.flow_user_trace_purchase_da
    WHERE length(query) > 0 AND length(query) < 50
    GROUP BY query, cpid, gender
    ORDER BY cnt DESC, cpid) as u INNER JOIN (SELECT pid, pno, lower(pname) as name, catid1, catid2, catid3
    FROM jiayundw_dm.product_profile_df
    WHERE date_id = '{profile_date}'
    GROUP BY pid, pno, name, catid1, catid2, catid3) as p
    ON u.cpid = p.pid""".format(profile_date=profile_date))
    df_3 = df_3.repartition(1000)

    # df_3 = df_3.select("query", "name")

    df_4 = spark.sql("""SELECT u.query, p.name FROM
    (SELECT lower(query) as query, pid as cpid, count(*) as cnt, gender FROM jiayundw_dwd.flow_user_trace_click_da
    WHERE length(query) > 0 AND length(query) < 50
    GROUP BY query, cpid, gender
    ORDER BY cnt DESC, cpid) as u INNER JOIN (SELECT pid, pno, lower(pname) as name, catid1, catid2, catid3
    FROM jiayundw_dm.product_profile_df
    WHERE date_id = '{profile_date}'
    GROUP BY pid, pno, name, catid1, catid2, catid3) as p
    ON u.cpid = p.pid
    AND cnt > 50""".format(profile_date=profile_date))
    df_4 = df_4.repartition(1000)
    # df_4 = df_4.select("query", "name")

    df = df.union(df_3).union(df_4)
    df = df.withColumn("label", functions.lit(1))
    df = df.repartition(2000)

    other_df = df.select('query')
    other_df = other_df.withColumn("name", choice_udf(df['query']))\
        .withColumn("label", functions.lit(0))
    other_df_1 = df.select('query')
    other_df_1 = other_df_1.withColumn("name", choice_udf(df['query']))\
        .withColumn("label", functions.lit(0))
    other_df_2 = df.select('query')
    other_df_2 = other_df_2.withColumn("name", choice_udf(df['query']))\
        .withColumn("label", functions.lit(0))
    other_df_3 = df.select('query')
    other_df_3 = other_df_3.withColumn("name", choice_udf(df['query']))\
        .withColumn("label", functions.lit(0))

    df = df.union(other_df).union(other_df_1).union(other_df_2).union(other_df_3)
    df = df.repartition(2000)
    df = df.withColumn('premise', sentence_map_udf(df['query'])).\
        withColumn('premise_mask', sentence_len_udf(df['query'])).\
        withColumn('hypothesis', sentence_map_udf(df['name'])).\
        withColumn('hypothesis_mask', sentence_len_udf(df['name']))
    df = df.drop('query', 'name')

    # df = df.fillna(0)
    # df.coalesce(500).write.mode("overwrite").options(header="true").csv(path=path)
    df = df.orderBy(rand())
    df.coalesce(500).write.format("tfrecords").option("recordType", "Example")\
        .option("codec", "org.apache.hadoop.io.compress.GzipCodec").save(train_path)

    df.limit(10000).coalesce(500).write.format("tfrecords").option("recordType", "Example")\
        .option("codec", "org.apache.hadoop.io.compress.GzipCodec").save(eval_path)

