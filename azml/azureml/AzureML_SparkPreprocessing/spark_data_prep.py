from argparse import ArgumentParser
from azureml.core import Run, Dataset, Datastore
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as f
from pyspark.sql.functions import monotonically_increasing_id, explode, lit, asc
from pyspark.sql.types import *


parser = ArgumentParser()
parser.add_argument('--train_input')
parser.add_argument('--input')
parser.add_argument('--process')
parser.add_argument('--output_dir')

args = parser.parse_args()

run = Run.get_context()
spark = SparkSession.builder.getOrCreate()

# Input Data Train/Test
raw_data = Dataset.get_by_id(run.experiment.workspace, id=args.input)

###### Bag of Words Model ######
### Data Preprocessing ###

if args.process == 'Training':

    # ==== Bag of Words - Preprocessing ==== #

    # Transform Train csv input to spark dataframe
    data = raw_data.to_spark_dataframe()

    # Get Item Description column
    corpus = data.select(f.collect_list('Item_Description')).first()[0]

    # Create Spark Dataframe
    doc = spark.createDataFrame(corpus, StringType())
    doc = doc.select('value').withColumn(
        "index", f.row_number().over(Window.orderBy(monotonically_increasing_id()))
    ).select(['index', 'value'])
    doc = doc.withColumnRenamed("value", "sentence")

    # Tokenize each corpus
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")

    tokenized = tokenizer.transform(doc)
    tokenized = tokenized.select("index", "sentence", "words")

    # Remove Stopwords
    remover = StopWordsRemover(inputCol="words", outputCol="filteredWords")
    final_tokenized = remover.transform(tokenized).select("index", "sentence", "filteredWords")

    # Remove numbers, special characters and whitespaces
    explode_df = final_tokenized.withColumn('word', explode(final_tokenized.filteredWords)).select(['index', 'word'])
    explode_df = explode_df.withColumn('word', f.regexp_replace('word', '[0-9]+', ''))
    explode_df = explode_df.withColumn('word', f.regexp_replace('word', '[^A-Za-z0-9]+', ''))
    explode_df = explode_df.filter("word <> ''")

    # ==== Create Training DTM ==== #

    # The minimum number of different documents a term must appear in to be included in the vocabulary
    minDF = 2

    counts = explode_df.groupBy(['index', 'word']).count()

    # Columns to drop based on minDF
    temp = counts.groupBy(['word']).sum('count')
    temp = temp.filter(f"sum(count) < '{minDF}'")
    columns_to_drop = list(sorted(temp.select(f.collect_list('word')).first()[0]))

    train_dtm = counts.groupBy(['index']).pivot('word').sum('count').na.fill(0).sort(asc("index"))
    train_dtm = train_dtm.drop(*columns_to_drop)

    # Create Category L4 dataframe to join with training dtm
    categories_df = data.select('Category_L4').withColumn(
        "index", f.row_number().over(Window.orderBy(monotonically_increasing_id()))
    )

    # Create final Train DTM dataframe
    final_df = train_dtm.join(categories_df, on='index', how='inner')
    final_df = final_df.drop('index')

    # Log number of features
    run.log("Number of features", len(final_df.columns[:-1]))

    # Write Training DTM csv
    final_df.coalesce(1).write \
        .mode('overwrite') \
        .option("header", "true") \
        .csv(args.output_dir)


else:

    # ==== Bag of Words - Preprocessing ==== #

    # Retrieve Train DTM
    train_dtm = spark.read.option("header", "true").csv(args.train_input)

    # Transform Inference csv input to spark dataframe
    data = raw_data.to_spark_dataframe()

    # Get Item Description column
    corpus = data.select(f.collect_list('Item_Description')).first()[0]

    # Create Spark Dataframe
    doc = spark.createDataFrame(corpus, StringType())
    doc = doc.select('value').withColumn(
        "index", f.row_number().over(Window.orderBy(monotonically_increasing_id()))
    ).select(['index', 'value'])
    doc = doc.withColumnRenamed("value", "sentence")

    # Tokenize each corpus
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")

    tokenized = tokenizer.transform(doc)
    tokenized = tokenized.select("index", "sentence", "words")

    # Remove Stopwords
    remover = StopWordsRemover(inputCol="words", outputCol="filteredWords")
    final_tokenized = remover.transform(tokenized).select("index", "sentence", "filteredWords")

    # Remove numbers, special characters and whitespaces
    explode_df = final_tokenized.withColumn('word', explode(final_tokenized.filteredWords)).select(['index', 'word'])
    explode_df = explode_df.withColumn('word', f.regexp_replace('word', '[0-9]+', ''))
    explode_df = explode_df.withColumn('word', f.regexp_replace('word', '[^A-Za-z0-9]+', ''))
    explode_df = explode_df.filter("word <> ''")

    # ==== Create Inference DTM ==== #

    # The minimum number of different documents a term must appear in to be included in the vocabulary
    minDF = 2

    counts = explode_df.groupBy(['index', 'word']).count()

    # Columns to drop based on minDF
    temp = counts.groupBy(['word']).sum('count')
    temp = temp.filter(f"sum(count) < '{minDF}'")
    columns_to_drop = list(sorted(temp.select(f.collect_list('word')).first()[0]))

    inference_dtm = counts.groupBy(['index']).pivot('word').sum('count').na.fill(0).sort(asc("index"))
    inference_dtm = inference_dtm.drop(*columns_to_drop)
    inference_dtm = inference_dtm.drop('index')

    train_cols = train_dtm.columns[:-1]  # exclude Category_L4 column
    inference_cols = inference_dtm.columns

    not_in_inference = list(set(train_cols) - set(inference_cols))
    extra_in_inference = list(set(inference_cols) - set(train_cols))

    print("Not in inference: ", len(not_in_inference))
    print("Extra in inference: ", len(extra_in_inference))

    # Dropping columns not present in inference DTM
    inference_dtm = inference_dtm.drop(*extra_in_inference)

    # Adding columns with zeros present in train and not in test
    if len(not_in_inference) != 0:
        for column in not_in_inference:
            inference_dtm = inference_dtm.withColumn(column, lit(0).cast('bigint'))

    inference_dtm = inference_dtm.select(train_cols)

    # Add index column to inference dtm
    inference_dtm = inference_dtm.withColumn(
        "index", f.row_number().over(Window.orderBy(monotonically_increasing_id()))
    )

    # Create Transactions dataframe
    transactions_df = data.select('Transaction_ID').withColumn(
        "index", f.row_number().over(Window.orderBy(monotonically_increasing_id()))
    )

    # Final inference dtm
    final_inf_df = inference_dtm.join(transactions_df, on='index', how='inner')
    final_inf_df = final_inf_df.drop('index')

    # Write Inference DTM csv
    final_inf_df.coalesce(1).write \
        .mode('overwrite') \
        .option("header", "true") \
        .csv(args.output_dir)
