from collections import Counter
from contextlib import closing
from flask import Flask, request, jsonify
import math
from collections import Counter
import inverted_index_colab
import hashed_index
from contextlib import closing
import re
from nltk.corpus import stopwords
import requests


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):

        self.N = 6348910
        self.body_index_path = '/content/body_index'
        self.title_index_path = '/content/title_index'
        self.anchor_index_path = '/content/anchor_index'
        self.pr_path = '/content/pr/'
        self.pv_path = '/content/pv/'
        self.id_name_path = '/content/id_name/'
        self.id_len_path = '/content/id_len/'

        # body_index_path = 'drive/MyDrive/Test Data/body_index'
        # title_index_path = 'drive/MyDrive/Test Data/title_index'
        # anchor_index_path = 'drive/MyDrive/Test Data/anchor_index'
        # pr_path = 'drive/MyDrive/Test Data/pr/'
        # pv_path = 'drive/MyDrive/Test Data/pv/'
        # id_name_path = 'drive/MyDrive/Test Data/id_name/'
        # id_len_path = 'drive/MyDrive/Test Data/id_len/'
        self.index_body = inverted_index_colab.InvertedIndex.read_index(self.body_index_path, 'index_text')
        self.index_title = inverted_index_colab.InvertedIndex.read_index(self.title_index_path, 'index_title')
        self.index_anchor = inverted_index_colab.InvertedIndex.read_index(self.anchor_index_path, 'index_anchor')
        self.id_len_dict = {}
        self.id_name_dict = {}
        self.id_pr_dict = {}
        self.id_pv_dict = {}
        self.CALLED_BY = False

        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)

app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
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
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    SERVER_DOMAIN = request.host_url[7:]
    # BEGIN SOLUTION
    CALLED_BY = True

    # TODO: threading might be possible
    body_res = requests.get('http://' + SERVER_DOMAIN + 'search_body?query=' + query)
    title_res = requests.get('http://' + SERVER_DOMAIN + 'search_title?query=' + query)
    anchor_res = requests.get('http://' + SERVER_DOMAIN + 'search_anchor?query=' + query)

    id_ranking = Counter()
    for i in range(len(body_res)):
        id_ranking[body_res[i][0]] += 2/i

    # TODO: find weight for title and anchor
    for i in range(len(title_res)):
        id_ranking[title_res[i][0]] += 1

    for i in range(len(anchor_res)):
        id_ranking[anchor_res[i][0]] += 1

    ids = list(id_ranking.keys())
    requests.post('http://' + SERVER_DOMAIN + '/get_pagerank', json=ids)
    requests.post('http://' + SERVER_DOMAIN + '/get_pageview', json=ids)

    # TODO: find weight for pr and pv
    for id in ids:
        pr = app.id_pr_dict[id]
        pv = app.id_pv_dict[id]
        if pr <= 2:
            id_ranking[id] = id_ranking[id] * math.log2(pr)
        if pv <= 10:
            id_ranking[id] = id_ranking[id] * math.log10(pv)

    res = sorted(list(map(lambda x: (x, app.id_name_dict[x]), ids)), key=lambda x: x[0], reverse=True)
    # END SOLUTION

    app.id_len_dict.clear()
    app.id_name_dict.clear()
    app.id_pr_dict.clear()
    app.id_pv_dict.clear()
    CALLED_BY = False

    return jsonify(res)


@app.route("/search_body")
def search_body():
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
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    query = tokenize(query)
    query_word_count = Counter(query)
    similarities = Counter()

    for term in query:
        posting_list = read_posting_list(app.index_body, term, app.body_index_path)
        idf = math.log2(app.N / app.index_body.df[term])
        for id, fr in posting_list:
            if id not in app.id_len_dict:
                app.id_len_dict.update(hashed_index.get_dict(app.id_len_path, 'id_len', id))
            tf = (fr / app.id_len_dict[id])
            weight = tf * idf
            similarities[id] += (weight * query_word_count[term])

    res = similarities.most_common(100)
    for id, score in res:
        if id not in app.id_name_dict:
            app.id_name_dict.update(hashed_index.get_dict(app.id_name_path, 'id_name', id))

    res = list(map(lambda x: (x[0], app.id_name_dict[x[0]]), res))

    # END SOLUTION
    if not app.CALLED_BY:
        app.id_len_dict.clear()
        app.id_name_dict.clear()

    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.
        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.[]
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    posting_lists = get_posting_lists(app.index_title, query, base_dir=app.title_index_path)
    return jsonify(posting_lists)
    for id, value in posting_lists:
        if id not in app.id_name_dict:
            app.id_name_dict.update(hashed_index.get_dict(app.id_name_path, 'id_name', id))

    if not app.CALLED_BY:
        res = list(map(lambda x: tuple((x[0], app.id_name_dict[x[0]])), posting_lists))

    # END SOLUTION
    if not app.CALLED_BY:
        app.id_name_dict.clear()

    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
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
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    posting_lists = get_posting_lists(app.index_anchor, query, base_dir=app.anchor_index_path)
    for id, value in posting_lists:
        if id not in app.id_name_dict:
            app.id_name_dict.update(hashed_index.get_dict(app.id_name_path, 'id_name', id))

    if not app.CALLED_BY:
        res = list(map(lambda x: tuple((x[0], app.id_name_dict[x[0]])), posting_lists))

    # END SOLUTION
    if not app.CALLED_BY:
        app.id_name_dict.clear()

    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
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
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    for id in wiki_ids:
        if id not in app.id_pr_dict:
            app.id_pr_dict.update(hashed_index.get_dict(app.pr_path, 'pr', id))

    res = (list(map(lambda x: app.id_pr_dict[x], wiki_ids)))

    # END SOLUTION
    if not app.CALLED_BY:
        app.id_pr_dict.clear()

    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
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
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    for id in wiki_ids:
        if id not in app.id_pv_dict:
            app.id_pv_dict.update(hashed_index.get_dict(app.pv_path, 'pv', id))

    res = (list(map(lambda x: app.id_pv_dict[x], wiki_ids)))

    # END SOLUTION
    if not app.CALLED_BY:
        app.id_pv_dict.clear()

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


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
