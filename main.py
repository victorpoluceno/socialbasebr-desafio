# -*- coding: utf-8 -*-
import json

from pyspark import SparkContext
from pyspark.sql import SQLContext

from pyspark.ml.feature import StopWordsRemover
from pyspark.mllib.linalg.distributed import (CoordinateMatrix,
                                              IndexedRow, IndexedRowMatrix)
from pyspark.ml.feature import Word2Vec


SOCIALBASEBR_DATA_FILE = 'data/socialbasebr_tweets.json'
FILTER_DATA_FILE = 'data/tweets.json'

TARGET_DATA_SIZE = 100


def unique_id():
    i = 0
    while True:
        yield i
        i += 1

id_gen = unique_id()


def load_data(filename, size=5, language_filter=None):
    '''Load and return tweets texts'''
    data = []
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break

            try:
                structure = json.loads(line)
            except ValueError:
                continue

            # Ignore deleted tweets and implement language filter
            if 'delete' in structure or language_filter and structure['lang'] \
                    not in language_filter:
                continue

            data.append(structure['text'])

    return data[:size]


def get_all_data():
    target_data = load_data(SOCIALBASEBR_DATA_FILE, TARGET_DATA_SIZE)
    filter_data = load_data(FILTER_DATA_FILE, 200,
                            language_filter=['pt', 'en'])
    return target_data + filter_data


def column_similarities(df):
    mat = IndexedRowMatrix(df.select("id", "result").map(
        lambda row: IndexedRow(*row)))
    java_coordinate_matrix = mat._java_matrix_wrapper.call(
        "columnSimilarities")
    return CoordinateMatrix(java_coordinate_matrix)


def fit_and_transform(sql_context, data):
    documentdf = sql_context.createDataFrame(data, ["id", "text"])

    # Learn a mapping from words to Vectors.
    word2vec = Word2Vec(vectorSize=len(data), minCount=0,
                        inputCol="text",
                        outputCol="result")
    model = word2vec.fit(documentdf)
    matrix = column_similarities(model.transform(documentdf))
    return matrix


def main(sc):
    sql_context = SQLContext(sc)
    all_data = get_all_data()

    # Input data: Each row is a bag of words from a sentence or document.
    training_data = [(id_gen.next(), text.split(" ")) for text in all_data]
    documentdf = sql_context.createDataFrame(training_data, ["id", "text"])

    remover = StopWordsRemover(inputCol="text", outputCol="text_filtered")
    cleaned_document = remover.transform(documentdf)

    # Learn a mapping from words to Vectors.
    word2vec = Word2Vec(vectorSize=len(training_data),
                        inputCol="text_filtered",
                        outputCol="result")
    model = word2vec.fit(cleaned_document)
    matrix = column_similarities(model.transform(cleaned_document))

    # We use the size of the target data to filter only
    # products of target data to filter data and avoid
    # products of taret data to itself
    values = matrix.entries.filter(
        lambda x: x.j >= TARGET_DATA_SIZE and x.i < TARGET_DATA_SIZE).sortBy(
        keyfunc=lambda x: x.value, ascending=False).map(
        lambda x: x.j).distinct().take(100)

    training_data_index = dict(training_data)
    for position, item in enumerate(values):
        line = " ".join(training_data_index[int(item)])
        print('%d -> %s' % (position, line.encode('utf-8')))


if __name__ == '__main__':
    main(SparkContext("local", "Desafio SocialBase"))
