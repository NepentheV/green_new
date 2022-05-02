import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
from input import DataInput, DataInputTest
from model_final import Model
import scipy.sparse as sp

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

flags = tf.app.flags
FLAGS = flags.FLAGS

# environment
flags.DEFINE_string('gpu', '1', 'gpu')
# hyperparam
# general
flags.DEFINE_integer('hidden_units', 128, 'hidden_units')   # 128
flags.DEFINE_integer('numHist', 0, 'numHist')   # retain the lastest l behaior (if numhist==0, there is no truncation)
# gnn
flags.DEFINE_bool('item_emb', False, 'item_emb')
flags.DEFINE_integer('gnn_layer', 4, 'gnn_layer')
flags.DEFINE_string('gnn_method', 'gcn_att4_rn',
                    'gcn, gcn_att3, gcn_att3_rn, gcn_att4, gcn_att4_rn')  # best: gcn_att4_rn
# gcl
flags.DEFINE_bool('gcl', False, 'gcl')
flags.DEFINE_integer('gcl_ran_num', 3000, 'The number of random.for example, 3000.')
flags.DEFINE_float('gcl_w', 1, 'The weight of DGI obj.')

# user_emb
flags.DEFINE_bool('user_emb', False, 'user_emb')
flags.DEFINE_bool('gnn_user_emb', False, 'gnn_user_emb.')   # The gnn of user_emb is the same as item.
flags.DEFINE_bool('gcl_user_emb', False, 'gcl_user_emb.')   # The gcl parameter of user_emb is the same as item.

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
train_batch_size = 32
test_batch_size = 512

print('hidden_units:'+str(FLAGS.hidden_units))
print('item_emb(graph):'+str(FLAGS.item_emb))
print('gnn_layer:'+str(FLAGS.gnn_layer))
print('gnn_method:'+FLAGS.gnn_method)
print('gcl:'+str(FLAGS.gcl))
print('gcl_ran_num:'+str(FLAGS.gcl_ran_num))
print('gcl_w:'+str(FLAGS.gcl_w))
print('user_emb:'+str(FLAGS.user_emb))
print('gnn_user_emb:'+str(FLAGS.gnn_user_emb))
print('gcl_user_emb:'+str(FLAGS.gcl_user_emb))

if FLAGS.numHist==0:
    datas='../pkl_MT/dataset.pkl'
else:
    datas = '../pkl_MT/dataset'+str(FLAGS.numHist)+'.pkl'  #../dataset20.pkl

with open(datas, 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count = pickle.load(f)

best_auc = 0.0


def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d: d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0]  # noclick
        tp2 += record[1]  # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None


def _auc_arr(score):
    score_p = score[:, 0]
    score_n = score[:, 1]
    # print "============== p ============="
    # print score_p
    # print "============== n ============="
    # print score_n
    score_arr = []
    for s in score_p.tolist():
        score_arr.append([0, 1, s])
    for s in score_n.tolist():
        score_arr.append([1, 0, s])
    return score_arr


def _eval(sess, model):
    auc_sum = 0.0
    score_arr = []
    logloss_arr = []
    for _, uij in DataInputTest(test_set, test_batch_size):
        logloss, auc_, score_ = model.eval(sess, uij)
        score_arr += _auc_arr(score_)
        auc_sum += auc_ * len(uij[0])
        logloss_arr.append(logloss)
    test_gauc = auc_sum / len(test_set)
    Auc = calc_auc(score_arr)
    global best_auc
    if best_auc < test_gauc:
        best_auc = test_gauc
        model.save(sess, 'save_path/ckpt')

    logloss = sum(logloss_arr) / len(logloss_arr)
    return logloss, test_gauc, Auc


#################################################################
if FLAGS.numHist == 0:
    remaps = '../raw_data_MT/remap.pkl'
else:
    remaps = '../raw_data_MT/remap'+str(FLAGS.numHist)+'.pkl'   #../raw_data/remap20.pkl

with open(remaps, 'rb') as f:
    reviews_df = pickle.load(f)
print('start constructing adjcent table')

if FLAGS.numHist==0:
    adj="./adj/"
else:
    adj = "./adj"+"_"+str(FLAGS.numHist)+"/"  #./adj_20/

if not os.path.exists(adj):
    os.makedirs(adj)
if not os.path.exists(adj+"locs.npy"):
    cnt = 0
    locs = []
    weights = []
    dic = {}
    wu = np.zeros(item_count)

    start = time.time()
    for _, hist in reviews_df.groupby('reviewerID'):
        # if len(hist['asin'].tolist()) > FLAGS.numHist:
        #     hist = hist[-FLAGS.numHist:]
        pos_list = hist['asin'].tolist()
        for i in range(0, len(pos_list) - 1):
            if pos_list[i] == pos_list[i + 1]:
                continue

            if pos_list[i] in dic.keys():
                if pos_list[i + 1] in dic[pos_list[i]].keys():
                    weights[dic[pos_list[i]][pos_list[i + 1]]] += 1
                    weights[dic[pos_list[i + 1]][pos_list[i]]] += 1
                    continue
                else:
                    dic[pos_list[i]].update({pos_list[i + 1]: len(weights)})
                    wu[pos_list[i]] += 1
            else:
                dic.update({pos_list[i]: {pos_list[i + 1]: len(weights)}})
                wu[pos_list[i]] += 1

            if pos_list[i + 1] in dic.keys():
                if pos_list[i] in dic[pos_list[i + 1]].keys():
                    weights[dic[pos_list[i + 1]][pos_list[i]]] += 1
                    weights[dic[pos_list[i]][pos_list[i + 1]]] += 1
                    continue
                else:
                    dic[pos_list[i + 1]].update({pos_list[i]: len(weights) + 1})
                    wu[pos_list[i + 1]] += 1
            else:
                dic.update({pos_list[i + 1]: {pos_list[i]: len(weights) + 1}})
                wu[pos_list[i + 1]] += 1

            locs.append([pos_list[i], pos_list[i + 1], wu[pos_list[i]] - 1])
            locs.append([pos_list[i + 1], pos_list[i], wu[pos_list[i + 1]] - 1])
            weights.append(1)
            weights.append(1)
            dic.update({})

        cnt += 1
        if cnt % 10000 == 0:
            print('cnt->', cnt, end='   ')
            print('time per interval: ', time.time() - start)
            start = time.time()

    locs = locs + [[i, i, wu[i]] for i in range(item_count)]
    # nums = nums + [item_count*i+i for i in range(item_count)]
    weights = weights + [1 for i in range(item_count)]

    locs = np.array(locs)
    weights = np.array(weights)

    vals = locs[:, 1]
    size = np.array([item_count, np.max(wu) + 1])

    cnt1 = 0
    locs1 = []
    weights1 = []
    dic1 = {}
    wu1 = np.zeros(user_count)

    for _, hist in reviews_df.groupby('asin'):  # 'asin': 'reviewerID', 'unixReviewTime'
        pos_user_list = hist['reviewerID'].tolist()  # 0 0 1 2 3 4
        pos_user_list = sorted(pos_user_list)  # 0 0 1 2 3 4
        if len(pos_user_list) > 40:
            continue
        for i in range(0, len(pos_user_list) - 1):
            for j in range(i + 1, len(pos_user_list)):
                if pos_user_list[i] == pos_user_list[j]:
                    continue
                if pos_user_list[i] in dic1.keys():
                    if pos_user_list[j] in dic1[pos_user_list[i]].keys():
                        weights1[dic1[pos_user_list[i]][pos_user_list[j]]] += 1
                        weights1[dic1[pos_user_list[j]][pos_user_list[i]]] += 1
                        continue
                    else:
                        dic1[pos_user_list[i]].update({pos_user_list[j]: len(weights1)})
                        wu1[pos_user_list[i]] += 1
                else:
                    dic1.update({pos_user_list[i]: {pos_user_list[j]: len(weights1)}})
                    wu1[pos_user_list[i]] += 1

                if pos_user_list[j] in dic1.keys():
                    if pos_user_list[i] in dic1[pos_user_list[j]].keys():
                        weights1[dic1[pos_user_list[j]][pos_user_list[i]]] += 1
                        weights1[dic1[pos_user_list[i]][pos_user_list[j]]] += 1
                        continue
                    else:
                        dic1[pos_user_list[j]].update({pos_user_list[i]: len(weights1) + 1})
                        wu1[pos_user_list[j]] += 1
                else:
                    dic1.update({pos_user_list[j]: {pos_user_list[i]: len(weights1) + 1}})
                    wu1[pos_user_list[j]] += 1

                locs1.append([pos_user_list[i], pos_user_list[j], wu1[pos_user_list[i]] - 1])
                locs1.append([pos_user_list[j], pos_user_list[i], wu1[pos_user_list[j]] - 1])
                weights1.append(1)
                weights1.append(1)
                dic1.update({})

        cnt1 += 1
        if cnt1 % 10000 == 0:
            print('cnt1->', cnt1, end='   ')
            print('time per interval: ', time.time() - start)
            start = time.time()

    locs1 = locs1 + [[i, i, wu1[i]] for i in range(user_count)]
    weights1 = weights1 + [1 for i in range(user_count)]
    locs1 = np.array(locs1)
    weights1 = np.array(weights1)
    vals1 = locs1[:, 1]
    size1 = np.array([user_count, np.max(wu1) + 1])

    np.save(adj+"locs.npy", locs)
    np.save(adj+"vals.npy", vals)
    np.save(adj+"weights.npy", weights)
    np.save(adj+"size.npy", size)

    np.save(adj+"locs1.npy", locs1)
    np.save(adj+"vals1.npy", vals1)
    np.save(adj+"weights1.npy", weights1)
    np.save(adj+"size1.npy", size1)
else:
    print('loading adjcent table from adj folder')
    locs = np.load(adj+"locs.npy")
    vals = np.load(adj+"vals.npy")
    weights = np.load(adj+"weights.npy")
    size = np.load(adj+"size.npy")

    locs1 = np.load(adj+"locs1.npy")
    vals1 = np.load(adj+"vals1.npy")
    weights1 = np.load(adj+"weights1.npy")
    size1 = np.load(adj+"size1.npy")


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

# for items' A
sgc_w_mod = 1
if sgc_w_mod:
    adj = sp.csc_matrix((weights, (locs[:, 0], locs[:, 1])), shape=[item_count, item_count])
else:
    adj = sp.csc_matrix((np.ones_like(weights), (locs[:, 0], locs[:, 1])), shape=[item_count, item_count])

support = normalize_adj(adj)
# print(support)
idx, idy, val = sp.find(support)
full_indices = tf.cast(tf.stack([idx, idy], axis=1), tf.int64)
support_in_sp = tf.SparseTensor(full_indices, tf.cast(val, tf.float32),
                                tf.constant((item_count, item_count), dtype=tf.int64))

# for users' A
if sgc_w_mod:
    adj1 = sp.csc_matrix((weights1, (locs1[:, 0], locs1[:, 1])), shape=[user_count, user_count])
else:
    adj1 = sp.csc_matrix((np.ones_like(weights1), (locs1[:, 0], locs1[:, 1])), shape=[user_count, user_count])

support1 = normalize_adj(adj1)
# print(support)
idx1, idy1, val1 = sp.find(support1)
full_indices1 = tf.cast(tf.stack([idx1, idy1], axis=1), tf.int64)
support_in_sp1 = tf.SparseTensor(full_indices1, tf.cast(val1, tf.float32),
                                tf.constant((user_count, user_count), dtype=tf.int64))

print('ready')

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
# with tf.Graph().as_default() as sess:
    model = Model(user_count, item_count, cate_count, cate_list, support_in_sp, support_in_sp1)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    from pathlib import Path

    model.summary_writer = tf.summary.FileWriter(
        'runs/{}_{}'.format(Path(__file__).name.split('.')[0], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
        graph=sess.graph)
    # model.summary_writer = tf.summary.FileWriter('runs/{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), graph=sess.graph)

    print('logloss: %.4f\t test_gauc: %.4f\t test_auc: %.4f' % _eval(sess, model))
    # stats_graph(sess)
    sys.stdout.flush()
    lr = 0.001
    start_time = time.time()
    for _ in range(50):

        random.shuffle(train_set)

        epoch_size = round(len(train_set) / train_batch_size)
        loss_sum = 0.0
        trainlogloss_sum = 0.0
        for _, uij in DataInput(train_set, train_batch_size):
            trainlogloss, loss = model.train(sess, uij, lr)
            loss_sum += loss
            trainlogloss_sum += trainlogloss

            if model.global_step.eval() % 1000 == 0:
                start = time.time()
                logloss, test_gauc, Auc = _eval(sess, model)
                end = time.time()
                print('Epoch %d Global_step %d\tTrain_loss: %.4f\tTrainlogloss: %.4f\tlogloss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f\tInference_time: %.4f' %
                      (model.global_epoch_step.eval(), model.global_step.eval(), 
                       loss_sum / 1000, trainlogloss_sum / 1000, logloss, test_gauc, Auc, end-start))
                sys.stdout.flush()
                loss_sum = 0.0
                trainlogloss_sum = 0.0

            if model.global_step.eval() % 336000 == 0:
                lr *= 0.9

        print('Epoch %d DONE\tCost time: %.2f' %
              (model.global_epoch_step.eval(), time.time() - start_time))
        sys.stdout.flush()
        model.global_epoch_step_op.eval()

    print('best test_gauc:', best_auc)
    sys.stdout.flush()
