from pathlib import Path
import pickle


def bin_index_hash(id):
    return int(round(id / 27135))


def write_a_dictionary(id_value, folder_path, name):
    curr_index = bin_index_hash(id_value[0][0])
    index_dic = {}
    for id, value in id_value:
        if bin_index_hash(id) > curr_index:
            with open(Path(folder_path + name + str(curr_index) + '.pkl'), 'wb') as f:
                pickle.dump(index_dic, f)
            curr_index = bin_index_hash(id)
            index_dic = {}
            index_dic[id] = value
        else:
            index_dic[id] = value

    if(index_dic):
        with open(Path(folder_path + name + str(curr_index) + '.pkl'), 'wb') as f:
            pickle.dump(index_dic, f)


def get_value(folder_path, name, id):
    index = str(bin_index_hash(id))
    with open(Path(folder_path + name + index + '.pkl'), 'rb') as f:
        wid2pv = pickle.load(f)
    return wid2pv[id]


def get_dict(folder_path, name, id):
    index = str(bin_index_hash(id))
    with open(Path(folder_path + name + index + '.pkl'), 'rb') as f:
        wid2pv = pickle.load(f)
    return wid2pv