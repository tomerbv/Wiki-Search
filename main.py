import csv
import gzip
import pickle
import time
import Indexing.inverted_index_colab as inverted_index

from contextlib import closing


def test_page_view():
    t = time.time()
    path = 'D:/University/semster e/Data Restoration/Project/Final data/pv/pageviews-202108-user.pkl'
    with open(path, 'rb') as f:
        wid2pv = pickle.loads(f.read())
    data_items = wid2pv.items()
    data_list = list(data_items)
    print(len(data_list))
    print(f"total time: {time.time() - t}")

# test_page_view()

def test_page_rank():
    t = time.time()
    path = 'D:/University/semster e/Data Restoration/Project/Final data/pr_part-00000-5dd2d933-fcca-4eae-a434-4cfddf90e066-c000.csv.gz'
    page_rank = {}
    wiki_ids = [34258, 38424, 92222, 196789, 315141]
    with gzip.open(path, "rt") as csvFile:
        csvreader = csv.reader(csvFile)
        for row in csvreader:
            page_rank[int(row[0])] = (float(row[1]))


    res = list(map(lambda x: (x, page_rank[x]) if x in page_rank else (x, 0), wiki_ids))
    print(res)
    print(page_rank[int(row[0])])
    print(time.time() - t)
test_page_rank()