import pickle
import pandas as pd

def to_df(file_path):
  with open(file_path, 'r') as fin:
    df = {}
    i = 0
    for line in fin:
      df[i] = eval(line)
      i += 1
    df = pd.DataFrame.from_dict(df, orient='index')
    return df

reviews_df = to_df('../raw_data_MT/reviews_Movies_and_TV_5.json')
# print("reviews_df",reviews_df)
with open('../raw_data_MT/reviews.pkl', 'wb') as f:
  pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

meta_df = to_df('../raw_data_MT/meta_Movies_and_TV.json')
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)
# print("meta_df",meta_df)
with open('../raw_data_MT/meta.pkl', 'wb') as f:
  pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)
