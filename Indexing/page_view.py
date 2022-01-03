import pickle
from pathlib import Path


class page_view_reader:

    @staticmethod
    def read_from_disk(pv_clean):
        """
        :param pv_clean:
        pv_clean = f'{p.stem}.pkl'
        """
        # read in the counter
        with open(pv_clean, 'rb') as f:
          wid2pv = pickle.loads(f.read())
        return wid2pv