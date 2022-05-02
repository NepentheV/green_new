import pandas as pd
import random
import pickle
import numpy as np

random.seed(1234)


reviews_df = pd.read_csv("../raw_data_movielens/ml-20m/ratings.csv")

meta_df = pd.read_csv("../raw_data_movielens/ml-20m/movies.csv")
meta_df = meta_df[["movieId", "genres"]]
meta_df['genres'] = meta_df['genres'].map(lambda x: x.split('|')[-1])

print(reviews_df)
print(meta_df)
print(x)

def build_map(df, col_name):
  key = sorted(df[col_name].unique().tolist())
  m = dict(zip(key, range(len(key))))
  df[col_name] = df[col_name].map(lambda x: m[x])
  return m, key

asin_map, asin_key = build_map(meta_df, 'movieId')
cate_map, cate_key = build_map(meta_df, 'genres')
revi_map, revi_key = build_map(reviews_df, 'userId')
print(reviews_df)
print(meta_df)

user_count, item_count, cate_count, example_count =\
    len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, item_count, cate_count, example_count))
# print(reviews_df)
# print(meta_df)
# print(x)

meta_df = meta_df.sort_values('movieId')
meta_df = meta_df.reset_index(drop=True)
reviews_df['movieId'] = reviews_df['movieId'].map(lambda x: asin_map[x])
reviews_df = reviews_df.sort_values(['userId', 'timestamp'])
reviews_df = reviews_df.reset_index(drop=True)
# reviews_df = reviews_df[['userId', 'movieId', 'timestamp']]

cate_list = [meta_df['genres'][i] for i in range(len(asin_map))]
cate_list = np.array(cate_list, dtype=np.int32)

print("reviews_df", reviews_df)
print("cate_list", cate_list)
with open('../raw_data_movielens/remap.pkl', 'wb') as f:
  pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL) # uid, iid
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL) # cid of iid line
  pickle.dump((user_count, item_count, cate_count, example_count),
              f, pickle.HIGHEST_PROTOCOL)
#   pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)
