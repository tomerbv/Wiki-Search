# Wiki-Search
Wikipedia Search Engine for Data Retrieval Course:
#Introduction
in this project we create a search engine for a wikipedia corpus. We started out by using google colab to create the file inverted_index_gcp that will be the base of our engine.
we used preproccesed data that are indexes and dictionaries of data so our engine could access all the data in a fast way. this engine uses many data sources ,that each one
was evaluated and given a weight that determines how much that data source effects what wikipedia pages we return from the query. the engine is written in python and runs on the
google cloud platform.

## 
code structure and organization:

### Evaluation
a folder holding a google colab page which shows the evaluation of the data that holds page ranks and number of views since August of every document in our                        corpus  this helped us understand how much weight we want to give this factors in the calculation to return the results of the query

### Test
a folder that holds the google colab files that we used to test the engine before we ran it on google cloud platform. we first checked how are engine works with a small corpus of about 20,000 documents in contrast to the large corpus which holds about 6,3000,000 documnets

### Writers
a folder files that are not used to run the engine. this files hold google colab pages that show an idea we had of how to hold the data effiecently in the disk

### inverted_index_gcp
this file is the base of our engine and is a class that we use to create and use our indexes

### search_frontend 
this is the file that runs the engine. in this file we use all our data and methods to perform evaluations that when given a query will help us return the
most releveant wikipedia pages

