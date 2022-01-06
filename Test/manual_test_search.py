import csv
import gzip
import json
import math
import pickle
from collections import Counter
from pathlib import Path
import inverted_index_colab
import wikipedia
from contextlib import closing


def search(query):
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.
        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    if len(query) == 0:
        return res
    # BEGIN SOLUTION

    # END SOLUTION
    return res


def search_body(query):
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.
        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    N = 6348910
    if len(query) == 0:
        return res
    # BEGIN SOLUTION

    query_word_count = Counter(query.split())
    similarities = Counter()
    base_dir = 'drive/MyDrive/Test Data/'
    with open(base_dir + 'doc_info_index/id_title_len_dict.json') as json_file:
        dict_from_json = json.load(json_file)
    inverted_body = inverted_index_colab.InvertedIndex.read_index(base_dir + 'body_index', 'index_text')
    for term in query.split():
        posting_list = read_posting_list(inverted_body, term, base_dir + 'body_index')
        idf = math.log2(N/inverted_body.df[term])
        for id_tf_pair in posting_list:
            tf = (id_tf_pair[1]/dict_from_json[str(id_tf_pair[0])][1])
            weight = tf * idf
            similarities[id_tf_pair[0]] += (weight * query_word_count[term])


    '''if normalizing documents is neccessary'''
    # for doc_id, sim in similarities.items():
    #     similarities[doc_id] = (sim * (1/len(query)) * (1/dict_from_json[str(doc_id)][1]))



    # END SOLUTION
    return similarities.most_common(100)


def search_title(query):
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.
        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    if len(query) == 0:
        return res
    # BEGIN SOLUTION

    base_dir = 'drive/MyDrive/Test Data/'

    base_dir = 'drive/MyDrive/Test Data/'
    with open(base_dir + 'doc_info_index/id_title_len_dict.json') as json_file:
        dict_from_json = json.load(json_file)

    posting_lists = get_posting_lists(query, 'index_title', base_dir='drive/MyDrive/Test Data/title_index')
    # each element is (id,tf) and we want it to be --> (id,title)
    res = list(map(lambda x: tuple((x[0], dict_from_json[str(x[0])][0])), posting_lists))

    # END SOLUTION
    return res


def search_anchor(query):
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.
        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    if len(query) == 0:
        return res

    # BEGIN SOLUTION

    base_dir = 'drive/MyDrive/Test Data/'
    with open(base_dir + 'doc_info_index/id_title_len_dict.json') as json_file:
        dict_from_json = json.load(json_file)
    posting_lists = get_posting_lists(query, 'index_anchor', base_dir='drive/MyDrive/Test Data/anchor_index')
    # each element is (id,tf) and we want it to be --> (id,title)
    res = list(map(lambda x: tuple((x[0], dict_from_json[str(x[0])][0])), posting_lists))

    # END SOLUTION

    return res

def get_pagerank(wiki_ids):
    ''' Returns PageRank values for a list of provided wiki article IDs.
        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    if len(wiki_ids) == 0:
        return res
    # BEGIN SOLUTION
    page_rank = {}
    fileName = 'drive/MyDrive/Test Data/page_rank.csv.gz'
    with gzip.open(fileName, "rt") as csvFile:
        csvreader = csv.reader(csvFile)
        for row in csvreader:
            page_rank[int(row[0])] = (float(row[1]))

    res = list(map(lambda x: (x,page_rank[x]) if x in page_rank else (x,0), wiki_ids))
    res.sort(key=lambda x: x[1], reverse=True)

    # END SOLUTION
    # return (id, page rank)
    return res


def get_pageview(wiki_ids):
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.
        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    if len(wiki_ids) == 0:
        return res
    # BEGIN SOLUTION

    fileName = 'drive/MyDrive/Test Data/pv/pageviews-202108-user.pkl'
    with open(fileName, 'rb') as f:
        wid2pv = pickle.loads(f.read())
        for id in wiki_ids:
            res.append(wid2pv[id])

    res = list(map(lambda x: (x, wid2pv[x]) if x in wid2pv else (x, 0), wiki_ids))
    res.sort(key=lambda x: float(x[1]), reverse=True)

    # END SOLUTION
    # return (id, page view)
    return res


def get_posting_lists(query, index_name, base_dir=''):
    '''
    :param query: input query
    :param index_name: body/title/anchor_text
    :param base_dir: path to the XXX.bin file
    :return: posting list for this query, each element is tuple(wiki_id,query f)
    '''

    inverted_title = inverted_index_colab.InvertedIndex.read_index(base_dir, index_name)
    posting_lists = []
    for word in query.split():
        posting_list = read_posting_list(inverted_title, word, base_dir)
        posting_lists += posting_list

    res = Counter()
    for pair in posting_lists:
        res[pair[0]] += pair[1]

    return res.most_common()


TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


def read_posting_list(inverted, w, base_dir=''):
    with closing(inverted_index_colab.MultiFileReader()) as reader:
        try:
            locs = inverted.posting_locs[w]
            new_locs = [tuple((base_dir + '/' + locs[0][0], locs[0][1]))]
            b = reader.read(new_locs, inverted.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list
        except:
            return []

