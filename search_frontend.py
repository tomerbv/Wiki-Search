from flask import Flask, request, jsonify
import math
from collections import Counter
import inverted_index_gcp
import hashed_index
from contextlib import closing
import re
from nltk.corpus import stopwords
import nltk
import threading

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        ''' in run initialization we  load the relevant indexes to the RAM memory for faster access
            and better reusing of addresses and such.
        '''

        self.N = 6348910

        self.body_index_path = 'postings_gcp'
        self.title_index_path = 'title_index'
        self.anchor_index_path = 'anchor_index'
        self.pr_path = 'pr/'
        self.pv_path = 'pv/'
        self.id_name_path = 'id_name/'
        self.id_len_path = 'id_len/'

        self.index_body = inverted_index_gcp.InvertedIndex.read_index(self.body_index_path, 'index')
        self.index_title = inverted_index_gcp.InvertedIndex.read_index(self.title_index_path, 'title_index')
        self.index_anchor = inverted_index_gcp.InvertedIndex.read_index(self.anchor_index_path, 'anchor_index')

        self.id_len_dict = hashed_index.get_all(app.id_len_path)
        self.id_name_dict = hashed_index.get_all(app.id_name_path)
        self.id_pr_dict = hashed_index.get_all(app.pr_path)
        self.id_pv_dict = hashed_index.get_all(app.pv_path)

        self.body_res = []
        self.title_res = []
        self.anchor_res = []

        self.stemmer = nltk.PorterStemmer()

        self.CALLED_BY = False

        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
nltk.download('stopwords')
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    '''
    search method is multithreaded and uses customed weights on the results of the different search methods
    to return a final accumulated result.
    first it sets CALLED_BY to True and by doing so lets the other search methods know they are utility methods
    and by that should not return any value, only update the global values.
    later it runs threads for each search method and iterates through their results, weighting them accordingly.
    lastly it weights in the page rank and page view, clears the global results and returns the final result.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION


    app.CALLED_BY = True

    body_thread = threading.Thread(search_BM25(),args=query)
    body_thread.start()
    title_thread = threading.Thread(search_title(), args=query)
    title_thread.start()
    anchor_thread = threading.Thread(search_anchor(),args=query)
    anchor_thread.start()

    id_ranking = Counter()
    body_thread.join()
    for i in range(len(app.body_res)):
        id_ranking[app.body_res[i][0]] += 120/((2*i)+1)

    title_thread.join()
    for i in range(len(app.title_res)):
        id_ranking[app.title_res[i][0]] += 60/((6*i)+1)

    anchor_thread.join()
    for i in range(len(app.anchor_res)):
        id_ranking[app.anchor_res[i][0]] += 30/((6*i)+1)

    ids = list(id_ranking.keys())

    for id in ids:
        try:
            pr = math.sqrt(app.id_pr_dict[id])
        except:
            pr = 1
        try:
            pv = math.sqrt(app.id_pv_dict[id])
        except:
            pv = 1
        if pr > 10:
            id_ranking[id] = id_ranking[id] * math.log10(pr)
        if pv > 10:
            id_ranking[id] = id_ranking[id] * math.log2(pv)

    for id, value in id_ranking.most_common(100):
        try:
            res.append((id, app.id_name_dict[id]))
        except:
            continue

    # END SOLUTION
    app.body_res = []
    app.title_res = []
    app.anchor_res = []

    app.CALLED_BY = False

    return jsonify(res)


def search_BM25(query=None):
    ''' exactly the same as search body method only the similarity method used here is BM25
        and not the classic tf-idf and cosine similarity.
        the method is a utility method for search.
    '''
    res = []
    if query is None:
        query = request.args.get('query', '')
    if len(query) == 0:
        if not app.CALLED_BY:
            return jsonify(res)
        return res
    # BEGIN SOLUTION

    query = tokenize(query)
    similarities = Counter()

    b = 0.75
    k1 = 1.2
    avgpl = 320
    for term in query:
        posting_list = read_posting_list(app.index_body, term, app.body_index_path)
        if posting_list:
            df = app.index_body.df[term]
            idf = math.log((app.N - df + 0.5) / (df + 0.5) + 1)
            for id, tf in posting_list:
                if id in app.id_len_dict:
                    pl = app.id_len_dict[id]
                    similarities[id] += idf * ((tf * (k1 + 1) / (tf + k1 * (1 - b + (b * (pl / avgpl))))))

    most = similarities.most_common(100)

    for id, value in most:
        try:
            res.append((id, app.id_name_dict[id]))
        except:
            continue

    # END SOLUTION
    if not app.CALLED_BY:
        return jsonify(res)

    app.body_res = res
    return



@app.route("/search_body")
def search_body(query=None):
    ''' this method tokenizes the query first. then it iterates over the queries and with the posting list of each
        term and it calculates it cosign similarity for each relevant document.
        then it returns the 100 highest scoring documents for all the terms in the query.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    if query is None:
        query = request.args.get('query', '')
    if len(query) == 0:
        if not app.CALLED_BY:
            return jsonify(res)
        return res
    # BEGIN SOLUTION

    query = tokenize(query)
    query_word_count = Counter(query)
    similarities = Counter()

    for term in query:
        posting_list = read_posting_list(app.index_body, term, app.body_index_path)
        idf = math.log2(app.N / app.index_body.df[term])
        for id, fr in posting_list:
            if id in app.id_len_dict:
                pl = app.id_len_dict[id]
                tf = (fr/pl)
            else:
                tf = 0
            weight = tf * idf
            similarities[id] += (weight * query_word_count[term])

    most = similarities.most_common(100)

    for id, value in most:
        try:
            res.append((id,app.id_name_dict[id]))
        except:
            continue

    # END SOLUTION
    if not app.CALLED_BY:
        return jsonify(res)

    app.body_res = res
    return


@app.route("/search_title")
def search_title(query=None):
    ''' this method tokenizes the query first. then it iterates over the queries and with the posting list of each
        term and rturns the 100 highest scoring documents for all the terms in the query.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    if query is None:
        query = request.args.get('query', '')
    if len(query) == 0:
        if not app.CALLED_BY:
            return jsonify(res)
        return res
    # BEGIN SOLUTION

    query = tokenize(query)
    for word in query:
        stemmed = app.stemmer.stem(word)
        if stemmed not in query:
            query.append(stemmed)

    posting_lists = get_posting_lists(app.index_title, query, base_dir=app.title_index_path)

    for id, value in posting_lists:
        try:
            res.append((id,app.id_name_dict[id]))
        except:
            continue

    # END SOLUTION
    if not app.CALLED_BY:
        return jsonify(res)

    app.title_res = res
    return


@app.route("/search_anchor")
def search_anchor(query=None):
    ''' this method tokenizes the query first. then it iterates over the queries and with the posting list of each
        term and rturns the 100 highest scoring documents for all the terms in the query.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    if query is None:
        query = request.args.get('query', '')
    if len(query) == 0:
        if not app.CALLED_BY:
            return jsonify(res)
        return res
    # BEGIN SOLUTION

    query = tokenize(query)
    for word in query:
        stemmed = app.stemmer.stem(word)
        if stemmed not in query:
            query.append(stemmed)

    posting_lists = get_posting_lists(app.index_anchor, query, base_dir=app.anchor_index_path)
    for id, value in posting_lists:
        try:
            res.append((id,app.id_name_dict[id]))
        except:
            continue

    # END SOLUTION
    if not app.CALLED_BY:
        return jsonify(res)

    app.anchor_res = res
    return


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank(wiki_ids=None):
    ''' this method iterates over the id's and with the page rank dictionary it returns the
        pairing page rank for all the terms in the query.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    if wiki_ids is None:
        wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        if not app.CALLED_BY:
            return jsonify(res)
        return res
    # BEGIN SOLUTION

    for id in wiki_ids:
        try:
            res.append(app.id_pr_dict[id])
        except:
            continue

    # END SOLUTION

    return jsonify(res)




@app.route("/get_pageview", methods=['POST'])
def get_pageview(wiki_ids=None):
    ''' this method iterates over the id's and with the page view dictionary it returns the
        pairing page view for all the terms in the query.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    if wiki_ids is None:
        wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        if not app.CALLED_BY:
            return jsonify(res)
        return res
    # BEGIN SOLUTION

    for id in wiki_ids:
        try:
            res.append(app.id_pv_dict[id])
        except:
            continue

    # END SOLUTION

    return jsonify(res)



def get_posting_lists(inverted_index, query, base_dir=''):
    '''
    :param query: input query
    :param index_name: body/title/anchor_text
    :param base_dir: path to the XXX.bin file
    :return: posting list for this query, each element is tuple(wiki_id,query f)
    '''

    posting_lists = []
    for word in query:
        posting_list = read_posting_list(inverted_index, word, base_dir)
        posting_lists += posting_list

    res = Counter()
    for pair in posting_lists:
        res[pair[0]] += pair[1]

    return res.most_common()


TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


def read_posting_list(inverted, w, base_dir=''):
    with closing(inverted_index_gcp.MultiFileReader()) as reader:
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


def tokenize(query):
    """
    :param query: string - the query provided
    :return: list of tokens after removing stopwords and punctuation
    """
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    return [token for token in tokens if token not in all_stopwords]


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
