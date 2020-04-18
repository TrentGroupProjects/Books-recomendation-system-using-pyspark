# Books-recomendation-system-using-pyspark
1. Project introduction
Recommendation systems are a big part of machine learning applications but also because they are related to understanding people. Therefore, we have decided to build a recommendation system for books. Using our big data and data mining knowledge, we will suggest users the books according to the book and rating information.
The research: Book management system on the cloud
1. we can store the big data
2. we can apply DML efficiently
3. we can make some EDA
4. we can apply high-performance maching learning methods
2. Involved tools and environment
2.1 Databases --- MongoDB
MongoDB is open source, document-based database. It uses JSON like documents with schema. Provides high availability, high performance and easy scalability. We have used MongoDB due to its flexible schema and performance.
2.2 Big data framework --- Apache Spark
Apache Spark is fast cluster computing system used for big data. It provides support for Java, Scala, Python, R and SQL. Its is a lightning fast data processing framework.
•	Spark SQL
Spark SQL is a component of Spark Core for structured data processing. We can easily execute SQL queries using Spark SQL on Spark Dataframes.
•	MLlib
MLlib is a machine learning component of Spark Core. It offers machine learning algorithms such as classification, clustering and regression etc.
2.3 Developing environment
•	Databricks
Databricks is a company founded by the original creators of Apache Spark. It provides platform to work with Spark with automated cluster management and IPython-style notebooks.
•	PySpark
PySpark is a Python API for Apache Spark. It provides large number of features for machine learning and data science.
3. The approach overview
MongoDB Atlas: We used MongoDB Atlas cloud service as our database provider to store our dataset. This served as our data source.  And after processing the data, we will push our results to the MongoDB database.
Apache Spark: For the data processing framework, we used Apache spark with Python (Pyspark). 
We applied some queries on our dataset by using Spark SQL 
We applied machine learning processing and clustering on our dataset by using Spark machine learning libraries

 
 
4. Dataset description
4.1 Data source
Our dataset is the book information on the “best books ever” list, scraped from Goodreads (the world’s largest site for readers and book recommendations).
The webpages of the booklist: https://www.goodreads.com/list/show/1.Best_Books_Ever
4.2 Structures
The dataset contains a table with 54300 rows (book records) and 12 columns (the attributes).	
attributes	Data type	Description
Book_title	String	Name of the book
Book_isbn	Decimal	ISBN of the book, if found on the Goodreads page
Book_author	String	The author(s) of the book, multiple authors are separated by the symbol '|'
genres	String	Genres that the book belongs to; this is user-provided information
Book_desc	String	A description of the book, as found on the Goodreads web page of the book.
Book_edition	String	The edition of the book
Book_format	String	Format of the book, i.e., hardcover, paperback, ebook, audio, etc.
Book_pages	String	Number of pages
Book_rating	Decimal	Average rating given by users
Book_rating_count	Integer	Number of ratings given by users
Book_review_count	Integer	Number of reviews given by users
Image_url	URL	URL of the book cover image


5. Importing data into MongoDB
As we are using MongoDB Atlas which is a cloud service, we have to load data into MongoDB atlas first. We have created cluster on atlas and imported data through our local system using python.
 
After running this code, we get records in our collection created in mongodb atlas.
 
Its better to use a cloud service for our project due to its speed and connectivity. Now for using data in our project we have to load data into our databricks cluster which we’ll show you in next step.
6. Loading data to spark
Now for spark we are using databricks cloud service. We have created cluster on databricks and added below configurations in our spark.
 
spark.mongodb.input.uri mongodb+srv://MyMongoDBUser:********@cluster0-ueaju.azure.mongodb.net/Books.BooksData?retryWrites=true&w=majority 
spark.mongodb.output.uri mongodb+srv://MyMongoDBUser:********@cluster0-ueaju.azure.mongodb.net/Books.BooksData?retryWrites=true&w=majority spark.databricks.delta.preview.enabled true
These tell our spark the input URI and output URI of our MongoDB Atlas cluster, database(Books) and collection(BooksData).
Then we installed mongoDB spark connector on our Databricks cluster by adding below Maven Library.
 
After setting all parameters we are ready to excecute command so that we can fetch our data from mongodb atlas.

 
Now in this spark command we are using DefaultSource which will use the default parameters set by us earlier and this will fetch our data from our mongodb atlas cluster.
 
Now we can see that our data is fetched from mongodb atlas cluster and we can proceed further.
Just before going forward we want discuss with you about the libraries we have installed in our databricks spark cluster. These are few additional libraries which will be useful in our code in future.  
7. applying DML operations
7.1 steps to run and display SQL queries in spark environment
SQL statements must be run against a table. Create a table that is a pointer to the DataFrame. This command will create a temporary table on which we’ll applying SQL queries using spark SQL functionality. 
BooksDf.createOrReplaceTempView("BooksData")
The format and syntax of the query is exactly same as SQL query. Now we’ll be applying some spark sql queries to get some interesting results.
Note: we are using toPandas() function to show our result in a good format.
Query 1: We are getting Top rated books using spark sql.
 
Query 2: We are getting 10 top rated authors who have written at least 3 books.
 
Query 3: In this query we getting most famous genres.
 

Query 4: For this we are fetching popular authors.
  
Now after using queries we’ll go one step further and to get better insights of our data we’ll start visualizing our data.
7.2 Basic visualization
To visualizing our data we’ll be using spark pyspark_dist_explore library.
1st : To begin with we are getting the frequency distribution of books ratings.
 
This shows us the rating distribution roughly follows a normal distribution and 95% of books rating are between 3.6 – 4.5 approximately.
2nd : For our second visualisation we are getting popular authors on the basis of number of reviews and books published by them.
 
3rd : We were also quiet interested to get corelations between few columns as it will be helpful in our further analysis. So we’ll be creating a correlation matrix between ratings, pages, review count and rating count. 
 
This correlation matrix with the help of heat map suggest us that rating count is strongly related to review count and rest of the factors are not affecting others much significantly. 
8.  applying machine learning --- book recommendation
8.1 Goals and objectives
Based on the booklist dataset, we constructed a recommendation machine to suggest books to users. The mechanism of the recommendation is content based, we applied clustering method on the book description to determine which books are close to each other, then recommend books in the same cluster. 
We would answer a sample question like: “What books can be recommended to a Harry Potter fan?”
The following work displayed the text processing and clustering. 
8.2 Tools and techniques
•	The spark environment
o	Spark v2.4.5
•	Programming language
o	Python
•	Libraries
o	python-nltk, pyspark.sql,  pyspark.ml
•	Standard preprocessing tools
o	tokenizer, StopWordsRemover, stemmer, normalizer
•	Text similarity measures
o	Term Frequency Inverse Document Frequency (TF-IDF) algorithm
•	Clustering model
o	k-means

8.3  Implementation and outputs
8.3.1 Preprocessing

•	split the text in to words
As it shown below, we sliced a piece of book description after above steps.
We found the book description is a chunk of text, which is hard to compare with other description chunks.
 
Thus, the critical step is to split the text into words, using words as single unit to vectorized and compute similarities later.
regex_tokenizer = RegexTokenizer(inputCol="book_desc",outputCol="book_desc_words",pattern="\\W")      # use RegexTokenizer() to create tokens
sdf_books = regex_tokenizer.transform(sdf_books)  # tranformation
 
•	remove stop words
Some words are used to make a sentence grammatically correct. They do not necessary give a lot of information. 
Top-10 stop words: ['i','me','my','myself','we','our','ours','ourselves','you','your']
It is necessary to remove stop words to remove irrelevant information
remover = StopWordsRemover(inputCol="book_desc_words", outputCol="words_filtered")
 

•	Stemming 
Stemming is a process of grouping together inflected forms of a word. 
e.g. for write/writes/wrote/written/writing, we want to turn them all into single word "write"
This process helped to decrease the workload of comparing between words and the replicated information in corpus.
# use stemmer from python library nltk, apply stem() on the pandas dataframe
stemmer = PorterStemmer()
pdf_books['words_stem'] = pdf_books.apply(lambda row: [stemmer.stem(token) for token in row['words_filtered']], axis=1)
 
note: an alternative way of stemming is to use the stemmer from Spark NLP library, which is a mature toolbox including many NLP features.
So far we completed some basic preprocessing on the book descriptions, the words length and its changes are shown below.
 

8.3.2 Data measuring and scaling --- TF-IDF
Now we got every book description as a series of words, in order to make it possible to measure the similarity/distance between books descriptions, fistly we have to find the most “representative” words in each description, when picking these important words, there are two things we have to consider:
o	The frequency of the word
The more frequent the word occurs, the more likely it is a key word;
o	The information the word contains
The more special information the word contains, the more features it provides to distinguish the text from others
Then we will use mathematical concepts to measure the frequency and speciality.
o	Term frequency TF(t,d) , computing the number of times that term t appears in document d; we use it to count the frequencies of the words in each book description
o	Document frequency DF(t,D), computing the number of documents that contains term t; we use it to count the number of book descriptions which contain a common word
o	Inverse document frequency IDF(t,D) is based on the DF(t,D), numerically measuring how much information a term provides
 
Combining the two factors together, that is the TF-IDF model, a feature vectorization method widely used in text mining to reflect the importance of a term to a document in the corpus.
 
To apply this model on our book descriptions, 
o	Firstly, we use HashingTF to hash the every piece of description into a feature vector
o	Then we use IDF to rescale the feature vectors; this generally improves performance when using text as features. Our feature vectors could then be passed to a learning algorithm.
# use hashingTF to get term frequency, then rescale the vector
hashingTF = 
HashingTF(inputCol="words_stem", outputCol="words_rawFeatures", numFeatures=20) 
featurizedData = hashingTF.transform(sdf_books) 
idf = IDF(inputCol="words_rawFeatures", outputCol="words_features") idfModel = idf.fit(featurizedData) 
rescaledData = idfModel.transform(featurizedData)
 

8.3.3 Data modelling --- K-means clustering
Clustering is a process to find groups of objects such that the objects in a group will be similar to one another, and different from the objects in other groups.
Based on our objective, we want to put the similar books in a group and reccommand relevent ones to readers. Thus, we will build a clustering model.

k-means is one of the most commonly used clustering algorithms that clusters the data points into a predefined number of clusters. It is a partitional clustering approach to cluster the items into independent k partitions, the k have to be defined in advance.

•	Find the proper k
Before the experiment, we have to decide how many clusters we expect, this is, to find the proper paramter k. That means we have to evaluation the performance of clustering prior to implement it.
Here, elbow curve is a good way to visualize the performance of different k. We applied elbow method to run k-means clustering on the dataset for k in range (1:20) and plot their distortions.
As it shown below, the kmeans worked from k = 3 and the distortions decreased by higher k values, considering the accuracy and efficience, we chose k = 10 to implement the following experiments.
 
•	Training and preditions

# Trains a k-means model, split all the books into 10 clusters 
kmeans = KMeans().setK(10).setSeed(1).setFeaturesCol("words_norm") 
model = kmeans.fit(normalizedData) 

# Make predictions, the clusters are respresented by numbers, from 0 to 9 clusters = model.transform(normalizedData)
We got the clusters shown as below, the column prediction contains the index of clusters the book belongs.  E.g. book “The Hunger Games” and  “Alices Adventures” are in the same cluster “0”.
 

•	Checking the density distribution of clusters 
 
•	Verification and evaluation
Back to our initial question, “What books can be recommended to a Harry Potter fan?”
We will follow the clustering machine to see what kinds of books will be recommend to the reader who likes the book 'Harry Potter and the Order of the Phoenix'
We found the its cluster is 9, then we filtered out all the books belong to the cluster 9.
 
The interesting thing we found is, although these algorithm-picked books can belong to various genres, most of them can be labelled as “Fantasy/Mystery/Young/Children/Fiction..”
Which partly match the features of Harry Potter book series.
Additonally, we filtered all books which title contains “Harry Potter” and to see if they are in a same cluster. 
 
9. output 
After getting our required results we’ll pushing our result into our mongodb database using command mentioned below.
 
After running this command we can see that our results are stored in our databases.
 
10. challenges and reflections
It’s quite natural to face issues while working on a project. We have also faced some issues during this project which we’ll discussing in this section.   
First challenge faced by us was MongoDB database issue as initially we thought of using MongoDB compass community as our database. But there were few connectivity issues due to which we’re not able to import our data into databricks spark cluster. Due to these issues we shifted to MongoDB atlas which was relatively easy and efficient.
 
Second thing which we are discussing its actually not a challenge rather a easy way of doing our work. We are talking about the visualisation tools offered by databricks spark cluster as for this we just only have to select the plot we want to use on what parameters using few clicks. So there is no need to code too much for visualization, which makes coding efficient.
 
Third challenge we want to discuss about is about spark functionality GraphX. Actually we first thought of using it in our project but unable to do it due to our project functionality. GraphX works best when social networking systems (where we have to do recommendations considering links and nodes). But in our project we are just clustering data and it was not efficient to use GraphX for our project.
Fourth challage faced was a problem faced due to restricted memory usuage provided to our databricks cluster. We know spark quite fast if we compare it to python as tested by us. We can see significant difference if we consider speeds between python and spark. But sometimes even using spark command excecution was quite slow due to restricted access provided to us. 
