import sys
import random

from loguru import logger
from py27hash import hash27

user_feats = ["userid", "gender", "age", "occupation"]
movie_feats = ["movieid", "title", "genres"]
rating_feats = ["userid", "movieid", "rating", "time"]

data_path = "ml-1m"
hash_dict = dict()
dict_size = 600000
test_user_path = "online_user"


def process(path: str):
    users_data = process_data(data_path+"/users.dat", user_feats)
    movie_data = process_movie_data(data_path+"movies.dat", movie_feats)

    for line in open(path, encoding="ISO-8859-1"):
        arr = line.strip().split("::")
        user_id = arr[0]
        movie_id = arr[1]
        out_line = f"time:{arr[3]}\t{users_data[user_id]}\t{movie_data[movie_id]}\tlabel:{arr[2]}"
        log_id = hash27(out_line) % 1000000000
        logger.info("%s %s", log_id, out_line)


def process_data(file_name: str, feats: list):
    processed = {}

    for line in open(file=file_name, encoding="ISO-8859-1"):
        arr = line.strip.split("::")
        out_line = ""
        for i in range(0, len(feats)):
            out_line += f"{feats[i]}:{arr[i]}\t"
        processed[arr[0]] = out_line.strip()

    return processed


def process_movie_data(file_name: str, feats: list):
    processed = {}

    for line in open(file=file_name, encoding="ISO-8859-1"):
        arr = line.strip().split("::")
        movie_title = ""
        movie_genres = ""

        for term in arr[1].split(" "):
            term = term.strip()
            if term != "":
                movie_title += f"{term} "

        for term in arr[2].split("|"):
            term = term.strip()
            if term != "":
                movie_genres += f"{term} "

        out_line = f"{feats[0]}:{arr[0]}\t{feats[1]}:{movie_title.strip()}\t{feats[2]}:{movie_genres.strip()}"
        processed[arr[0]] = out_line.strip()
    return processed


def get_hash(path: str):
    for line in open(path, encoding="ISO-8859-1"):
        arr = line.strip().split("\t")
        out_str = f"logid:{arr[0]} {arr[1]} {to_hash(arr[2])} {to_hash(arr[3])}  \
                           {to_hash(arr[4])} {to_hash(arr[5])} {to_hash(arr[6])} \
                            {to_hash_list(arr[7])} { to_hash_list(arr[8])} {arr[9]}"
        print(out_str)


def to_hash(in_str):
    feas = in_str.split(":")[0]
    arr = in_str.split(":")[1]
    out_str = "%s:%s" % (feas, (arr + arr[::-1] + arr[::-2] + arr[::-3]))
    hash_id = hash27(out_str) % dict_size
    if hash_id in hash_dict and hash_dict[hash_id] != out_str:
        print(hash_id, out_str, hash27(out_str))
        print("conflict")
        exit(-1)

    return "%s:%s" % (feas, hash_id)


def to_hash_list(in_str):
    arr = in_str.split(":")
    tmp_arr = arr[1].split(" ")
    out_str = ""
    for item in tmp_arr:
        item = item.strip()
        if item != "":
            key = "%s:%s" % (arr[0], item)
            out_str += "%s " % (to_hash(key))
    return out_str.strip()


def parse_data(file_name, feas):
    dict = {}
    for line in open(file_name, encoding='ISO-8859-1'):
        line = line.strip()
        arr = line.split("::")
        out_str = ""
        for i in range(0, len(feas)):
            out_str += "%s:%s\t" % (feas[i], arr[i])

        dict[arr[0]] = out_str.strip()
    return dict


def parse_movie_data(file_name, feas):
    dict = {}
    for line in open(file_name, encoding='ISO-8859-1'):
        line = line.strip()
        arr = line.split("::")
        title_str = ""
        genres_str = ""

        for term in arr[1].split(" "):
            term = term.strip()
            if term != "":
                title_str += "%s " % (term)
        for term in arr[2].split("|"):
            term = term.strip()
            if term != "":
                genres_str += "%s " % (term)
        out_str = "movieid:%s\ttitle:%s\tgenres:%s" % (
            arr[0], title_str.strip(), genres_str.strip())
        dict[arr[0]] = out_str.strip()
    return dict


def generate_online_user():
    movie_dict = parse_movie_data(data_path + "/movies.dat", movie_fea)

    with open(test_user_path + "/movies.dat", 'w') as f:
        for line in open(test_user_path + "/users.dat"):
            line = line.strip()
            arr = line.split("::")
            userid = arr[0]
            for item in movie_dict:
                f.write(userid + "::" + item + "::1")
                f.write("\n")


def generate_online_data(path):
    user_dict = parse_data(data_path + "/users.dat", user_feats)
    movie_dict = parse_movie_data(data_path + "/movies.dat", movie_feats)

    for line in open(path, encoding='ISO-8859-1'):
        line = line.strip()
        arr = line.split("::")
        userid = arr[0]
        movieid = arr[1]
        label = arr[2]
        out_str = "time:%s\t%s\t%s\tlabel:%s" % ("1", user_dict[userid],
                                                 movie_dict[movieid], label)
        log_id = hash27(out_str) % 1000000000
        res = "%s\t%s" % (log_id, out_str)
        arr = res.strip().split("\t")
        out_str = "logid:%s %s %s %s %s %s %s %s %s %s" % \
            (arr[0], arr[1], to_hash(arr[2]), to_hash(arr[3]), to_hash(arr[4]), to_hash(arr[5]),
             to_hash(arr[6]), to_hash_list(arr[7]), to_hash_list(arr[8]), arr[9])
        print(out_str)


if __name__ == "__main__":
    random.seed(1111111)
    if sys.argv[1] == "process_raw":
        process(sys.argv[2])
    elif sys.argv[1] == "hash":
        get_hash(sys.argv[2])
    elif sys.argv[1] == "data_recall":
        generate_online_user()
        generate_online_data(test_user_path + "/movies.dat")
    elif sys.argv[1] == "data_rank":
        generate_online_data(test_user_path + "/movies.dat")
