import math
from collections import Counter
import inverted_index_colab
import hashed_index
from contextlib import closing
import re
from nltk.corpus import stopwords

def tokenize(query):
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    return [token for token in tokens if token not in all_stopwords]



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

    query = tokenize(query)


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

    query = tokenize(query)

    query_word_count = Counter(query)
    similarities = Counter()
    body_index_path = 'drive/MyDrive/Test Data/body_index'
    id_len_path = 'drive/MyDrive/Test Data/id_len/'

    id_len_dict = {}
    bucket_access = [False for i in range(2521)]

    inverted_body = inverted_index_colab.InvertedIndex.read_index(body_index_path, 'index_text')
    for term in query:
        posting_list = read_posting_list(inverted_body, term, body_index_path)
        idf = math.log2(N/inverted_body.df[term])
        for id, fr in posting_list:
            if not bucket_access[hashed_index.bin_index_hash(id)]:
                id_len_dict.update(hashed_index.get_dict(id_len_path, 'id_len', id))
                bucket_access[hashed_index.bin_index_hash(id)] = True
            tf = (fr/id_len_dict[id])
            weight = tf * idf
            similarities[id] += (weight * query_word_count[term])


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

    query = tokenize(query)

    title_index_path = 'drive/MyDrive/Test Data/title_index'
    id_name_path = 'drive/MyDrive/Test Data/id_name/'

    posting_lists = get_posting_lists(query, 'index_title', base_dir=title_index_path)
    names_dict = {}
    bucket_access = [False for i in range(2521)]
    for i in posting_lists:
        if not bucket_access[hashed_index.bin_index_hash(i[0])]:
            names_dict.update(hashed_index.get_dict(id_name_path, 'id_name', i[0]))
            bucket_access[hashed_index.bin_index_hash(i[0])] = True

    # each element is (id,tf) and we want it to be --> (id,title)
    res = list(map(lambda x: tuple((x[0], names_dict[x[0]])), posting_lists))

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

    query = tokenize(query)

    anchor_index_path = 'drive/MyDrive/Test Data/anchor_index'
    id_name_path = 'drive/MyDrive/Test Data/id_name/'

    posting_lists = get_posting_lists(query, 'index_anchor', base_dir=anchor_index_path)
    names_dict = {}
    bucket_access = [False for i in range(2521)]
    for i in posting_lists:
        if not bucket_access[hashed_index.bin_index_hash(i[0])]:
            names_dict.update(hashed_index.get_dict(id_name_path, 'id_name', i[0]))
            bucket_access[hashed_index.bin_index_hash(i[0])] = True

    # each element is (id,tf) and we want it to be --> (id,title)
    res = list(map(lambda x: tuple((x[0], names_dict[x[0]])), posting_lists))

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

    pr_path = 'drive/MyDrive/Test Data/pr/'

    page_rank = {}
    bucket_access = [False for i in range(2521)]
    for id in wiki_ids:
        if not bucket_access[hashed_index.bin_index_hash(id)]:
            page_rank.update(hashed_index.get_dict(pr_path, 'pr', id))
            bucket_access[hashed_index.bin_index_hash(id)] = True

    res = sorted(list(map(lambda x: (x, page_rank[x]) , wiki_ids)),key=lambda x: x[1], reverse=True)

    # END SOLUTION
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

    pv_path = 'drive/MyDrive/Test Data/pv/'

    page_view = {}
    bucket_access = [False for i in range(2521)]
    for id in wiki_ids:
        if not bucket_access[hashed_index.bin_index_hash(id)]:
            page_view.update(hashed_index.get_dict(pv_path, 'pv', id))
            bucket_access[hashed_index.bin_index_hash(id)] = True

    res = sorted(list(map(lambda x: (x, page_view[x]), wiki_ids)), key=lambda x: x[1], reverse=True)

    # END SOLUTION
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
    for word in query:
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
