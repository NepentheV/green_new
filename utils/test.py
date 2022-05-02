import pickle


#   pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL) # uid, iid
#   pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL) # cid of iid line
#   pickle.dump((user_count, item_count, cate_count, example_count),
#               f, pickle.HIGHEST_PROTOCOL)
#   pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)

with open('../raw_data/remap.pkl', 'rb') as f:
    reviews_df = pickle.load(f, encoding='utf-8')
    cate_list = pickle.load(f, encoding='utf-8')
    user_count, item_count, cate_count, example_count = pickle.load(f, encoding='utf-8')
    asin_key, cate_key, revi_key = pickle.load(f, encoding='utf-8')


print("reviews_df", reviews_df)  #  reviewerID   asin  unixReviewTime
print("cate_list", cate_list)    # [738 157 571 ...  63 674 351]

print("num", len(cate_list))    # [738 157 571 ...  63 674 351]


# print("user_count", user_count)
# print("item_count", item_count)
# print("cate_count", cate_count)
# print("example_count", example_count)

# print("asin_key", asin_key)
# print("cate_key", cate_key)
# print("revi_key", revi_key)

