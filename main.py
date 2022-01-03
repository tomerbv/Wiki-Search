import pickle
from pathlib import Path



path = 'D:/University/semster e/Data Restoration/Project/Final data/pageviews-202108-user.pkl'
with open(path, 'rb') as f:
    wid2pv = pickle.loads(f.read())
data_items = wid2pv.items()
data_list = list(data_items)[:20]
print(data_list)