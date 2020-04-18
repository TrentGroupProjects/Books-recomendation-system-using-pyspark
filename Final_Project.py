# Databricks notebook source

# importing the necessary libraries
import numpy as np
from pyspark.sql import SQLContext
from pyspark.ml.feature import RegexTokenizer,StopWordsRemover
from nltk.stem.porter import *
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import Normalizer
from pyspark.ml.clustering import KMeans


# COMMAND ----------

import pandas as pd
BooksDf = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()
BooksDf.createOrReplaceTempView("BooksData")

# COMMAND ----------

#Using Saprk SQL
#Query 1: Get top rated Books
query='''SELECT book_title,book_authors,book_rating FROM BooksData ORDER BY book_rating DESC LIMIT 5'''
spark.sql(query).toPandas()

# COMMAND ----------

#Query 2: Get top rated authors who have written atleast 3 books.
query='''SELECT book_authors,AVG(book_rating) AS avg_rating, COUNT(*) AS book_count FROM BooksData GROUP BY book_authors HAVING book_count>3 ORDER BY avg_rating DESC LIMIT 10'''
df2=spark.sql(query).toPandas()
df2

# COMMAND ----------

#Query 3: Get most popular genres.
query='''SELECT genres, SUM(book_review_count) AS total_review_count,ROUND(AVG(book_rating)) AS avg_book_rating FROM BooksData GROUP BY genres ORDER BY total_review_count DESC LIMIT 5'''
df3=spark.sql(query).toPandas()
df3

# COMMAND ----------

from pyspark_dist_explore import Histogram, hist, distplot, pandas_histogram
import matplotlib.pyplot as plt
#Frequency graph for Book ratings
test_df_book_rating = BooksDf.select('book_rating')
display(test_df_book_rating)

# COMMAND ----------

#Query 4: displaying the popular authors 
query='''SELECT book_authors, sum(book_review_count) AS total_review_count,count(*) as books_written FROM BooksData GROUP BY book_authors ORDER BY total_review_count DESC LIMIT 5'''
df4=spark.sql(query).toPandas()
display(df4)

# COMMAND ----------

#Correlation between book pages, book rating, book rating count and book review count
BooksDf.toPandas()
import seaborn as sns
df8=BooksDf[['book_pages','book_rating','book_rating_count','book_review_count']]
df8=df8.toPandas()
df8.book_pages=df8.book_pages.str.replace(' pages','').str.replace(' page','')
df8['book_pages'] = pd.to_numeric(df8['book_pages'])
display(sns.heatmap(df8.corr(),annot=True))

# COMMAND ----------

#Data Pre-Processing
#Selecting interested columns
from pyspark.sql.functions import *
from pyspark.sql import functions as sf
BooksDf=BooksDf.withColumn('_id',sf.col('_id').cast("string"))
book_df=BooksDf
book_df=book_df[['_id','book_title','book_desc','book_authors','genres','book_rating']]
#Pre-Processing Part 1: Removing rows with NaN
book_df=book_df.dropna(how='any')
#Pre-Processing Part 2: Keeping only alphabets from all the columns
#df.withColumn('address', regexp_replace('address', 'lane', 'ln'))
book_df=book_df.withColumn('book_desc',regexp_replace('book_desc',"[^A-Za-z ]+", ""))
book_df=book_df.withColumn('book_title',regexp_replace('book_title',"[^A-Za-z ]+", ""))
book_df=book_df.withColumn('genres',regexp_replace('genres','\\|', ' '))
book_df=book_df.withColumn('genres',regexp_replace('genres',"[^A-Za-z ]+", ""))
#Pre-Processing Part 3: Removing rows with no Book description and no Book title (empty and spaces)
book_df=book_df.filter(~book_df["book_desc"].isin([""]))
book_df=book_df.withColumn('book_desc',regexp_replace('book_desc'," +"," "))
book_df=book_df.withColumn('book_title',regexp_replace('book_title'," +"," "))
book_df=book_df.filter(~book_df["book_desc"].isin([" "]))
book_df=book_df.filter(~book_df["book_title"].isin([" "]))
sdf_books = book_df
sdf_books.head()

# COMMAND ----------

# 2.1 splitting text into words
regex_tokenizer = RegexTokenizer(inputCol="book_desc", outputCol="book_desc_words", pattern="\\W")
sdf_books = regex_tokenizer.transform(sdf_books)
sdf_books.select("book_title", "book_desc", "book_desc_words").show()

# COMMAND ----------

# 2.2 removing stop words
remover = StopWordsRemover(inputCol="book_desc_words", outputCol="words_filtered")

# top-10 stop words ['i','me','my','myself','we','our','ours','ourselves','you','your']

# remove stopwords from the book description
sdf_books = remover.transform(sdf_books)
sdf_books.select("book_title", "book_desc", "book_desc_words", "words_filtered").show()

# COMMAND ----------

# 2.3 Stemming of the tokenized corpus
# Stemming is a process of grouping together inflected forms of a word. e.g. for write/writes/wrote/written/writing, We want to turn them all into single word "write"
# we will use the stemmer nltk (natural language tookkit) library
# Instantiate stemmer object
stemmer = PorterStemmer()

# it is more handy to use stemmer on the pandas dataframe, so we make a transformation and apply the stem()
pdf_books = sdf_books.toPandas()
pdf_books['words_stem'] = pdf_books.apply(lambda row: [stemmer.stem(token) for token in row['words_filtered']], axis=1)
pdf_books[["book_title", "book_desc", "book_desc_words", "words_filtered", "words_stem"]].head()


# COMMAND ----------


# so far we did some cleaning jobs, we can compare the length of the words before and after the processing
# count the length
pdf_books['words_len_before'] = pdf_books.apply(lambda row: len(row['book_desc_words']), axis=1) 
pdf_books['words_len_after'] = pdf_books.apply(lambda row: len(row['words_stem']), axis=1) 
pdf_books[['words_len_before','words_len_after']].head()

# COMMAND ----------

# 3. aftering the preprocessing the book description, we will implement data modelling

# 3.1 tf-idf
# use hashingTF to get term frequency, alternatively, countVectorizer has the same function
sdf_books = sqlContext.createDataFrame(pdf_books)
hashingTF = HashingTF(inputCol="words_stem", outputCol="words_rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(sdf_books)

# use idf to inverse frequency
idf = IDF(inputCol="words_rawFeatures", outputCol="words_features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData.select("book_title", "words_stem","words_features").show()

# COMMAND ----------


# 3.2 l2 normalize the feature vector
normalizer = Normalizer(inputCol="words_features", outputCol="words_norm")
normalizedData = normalizer.transform(rescaledData)

normalizedData.select("book_title", "words_stem","words_features","words_norm").show()

# COMMAND ----------

# 4. apply kmeans clustering method on the normalized book description words

# Trains a k-means model, split all the books into 10 clusters according the similarities between book descriptions
kmeans = KMeans().setK(10).setSeed(1).setFeaturesCol("words_norm")
model = kmeans.fit(normalizedData)

# Make predictions, the clusters are respresented by numbers, from 0 to 9
clusters = model.transform(normalizedData)
clusters.select("book_title", "prediction").show()

# COMMAND ----------

# plot to see the distribution of clusters
display(clusters)

# COMMAND ----------

# one of the Harry Potter series books belongs to one cluster, find the index of the cluster
clusters.filter(clusters.book_title == 'Harry Potter and the Order of the Phoenix').select("prediction").show()

# COMMAND ----------

# apply the kmeans model to find the recommended books which belongs to the same cluster as the target book
clusters.filter(clusters.prediction == 9).select("book_title", "genres").show()

# COMMAND ----------

# the most common recommondation is the books from a same series or written by a same author
# we filtered all the books which contain "Harry Potter" in the title, and we check if all of the books be clustered in a same group
clusters.registerTempTable('books')
target_book = sqlContext.sql("select distinct book_title, prediction from books where book_title LIKE '%Harry Potter%' AND prediction = 9")
target_book.show()

# COMMAND ----------

#Now we are pushing our results into mongodb 
target_book.write.format("com.mongodb.spark.sql.DefaultSource").mode("append").option("database","Books").option("collection","RecommendedBooks").save()

# COMMAND ----------

# find the optimal number of clusters (the value of k)
# plot Elbow curve after running kmeans algorithm for a range of k values (on x-axis) and distortion/cost (on y-axis)
import numpy as np
cost = np.zeros(20)
for k in range(3,20):
  kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("words_norm")
  model = kmeans.fit(normalizedData)
  cost[k] = model.computeCost(normalizedData)
plt.plot(cost, 'bx-')
plt.xlabel('number of clusters')
plt.ylabel('distortions of costs')
plt.title('Elbow Method For Optimal k')
plt.show()
