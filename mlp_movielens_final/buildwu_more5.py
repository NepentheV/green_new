import random
import pickle

random.seed(1234)

with open('../raw_data_movielens/remap.pkl', 'rb') as f:
    reviews_df = pickle.load(f)    # userId,movieId,rating,timestamp
    cate_list = pickle.load(f)
    user_count, item_count, cate_count, example_count = pickle.load(f)

train_set = []
test_set = []

print(reviews_df)
for reviewerID, hist in reviews_df.groupby('userId'):     # hist:'userId', 'movieId', 'rating', 'timestamp'
    pos_train_set = []
    neg_train_set = []
    pos_list = []
    neg_list = []
    for i in range(len(hist['rating'])):
        if hist['rating'].tolist()[i] >= 4:
            if len(pos_list) > 10:
                pos_train_set.append((reviewerID, pos_list[-10:], hist['movieId'].tolist()[i], 1))
            else:
                pos_train_set.append((reviewerID, pos_list, hist['movieId'].tolist()[i], 1))
        else:
            if len(pos_list) > 10:
                neg_train_set.append((reviewerID, pos_list[-10:], hist['movieId'].tolist()[i], 0))
            else:
                neg_train_set.append((reviewerID, pos_list, hist['movieId'].tolist()[i], 0))

        if hist['rating'].tolist()[i] >= 4:
            pos_list.append(hist['movieId'].tolist()[i])
        else:
            neg_list.append(hist['movieId'].tolist()[i])

    if len(pos_list)==0 or len(neg_list)==0 or len(pos_train_set)==0 or len(neg_train_set)==0:
        continue

    lenPos = len(pos_train_set) - 1   # len(pos_train_set)=10 lenpos=9
    lenNeg = len(neg_train_set)       # 10
    if lenPos > lenNeg:
        pos_train_set = pos_train_set[-(lenNeg+1):]
    if lenPos < lenNeg:
        neg_train_set = neg_train_set[-lenPos:]
    if len(pos_train_set)+len(neg_train_set) < 6:
        continue
    if len(pos_train_set)+len(neg_train_set) > 20:
        pos_train_set = pos_train_set[-11:]
        neg_train_set = neg_train_set[-10:]

    train_set.extend(neg_train_set)
    train_set.extend(pos_train_set)

    test = train_set.pop()
    test0 = test[0]  # reviewerID
    test1 = test[1]  # hist
    test2 = test[2]  # target
    test3 = test[3]  # y

    def gen(list):
        random.shuffle(list)
        return list[0]

    if test3 == 0:
        label = (gen(pos_list), test2)
    else:
        label = (test2, gen(neg_list))
    test_set.append((test0, test1, label))

random.shuffle(train_set)
random.shuffle(test_set)

print(len(train_set))
print(len(test_set))

# assert len(test_set) == user_count
# assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])
train_batch_size = 32
test_batch_size = 512
predict_batch_size = 32

with open('../pkl_movielens/wuwu10_5_1/train_set.pkl', 'wb') as f:
  i = 0
  while i * train_batch_size < len(train_set):
    # train_set = train_set[i * train_batch_size: min((i + 1) * train_batch_size, len(train_set))]
    pickle.dump(train_set[i * train_batch_size: min((i + 1) * train_batch_size, len(train_set))], f, pickle.HIGHEST_PROTOCOL)
    i = i + 1

print('1')

with open('../pkl_movielens/wuwu10_5_1/test_set.pkl', 'wb') as f:
  i = 0
  while i * test_batch_size < len(test_set):
    # test_set = test_set[i * test_batch_size: min((i + 1) * test_batch_size, len(test_set))]
    pickle.dump(test_set[i * test_batch_size: min((i + 1) * test_batch_size, len(test_set))], f, pickle.HIGHEST_PROTOCOL)
    i = i + 1

print('2')

with open('../pkl_movielens/wuwu10_5_1/predict_set.pkl', 'wb') as f:
  i = 0
  while i * predict_batch_size < len(test_set):
    pickle.dump(test_set[i * predict_batch_size: min((i + 1) * predict_batch_size, len(test_set))], f, pickle.HIGHEST_PROTOCOL)
    i = i + 1

print('3')

with open('../pkl_movielens/wuwu10_5_1/cate_list.pkl', 'wb') as f:
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
with open('../pkl_movielens/wuwu10_5_1/num.pkl', 'wb') as f:
  pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)



