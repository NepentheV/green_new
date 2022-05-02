import tensorflow as tf


# from din.Dice import dice

def attention_three_part_gcn_layer(fin0, fin1, activation=None, name='', reuse=None):
    fin_emb = tf.concat([fin0, fin1 - fin0, fin1], axis=1)
    return tf.layers.dense(fin_emb, 1, activation=activation, name=name, reuse=reuse)


def attention_four_part_gcn_layer(fin0, fin1, activation=None, name='', reuse=None):
    fin_emb = tf.concat([fin0, fin1 - fin0, fin1, fin0 * fin1], axis=1)   # n*4d
    return tf.layers.dense(fin_emb, 1, activation=activation, name=name, reuse=reuse)


def gcn_my_att_layer(support_in_sp, item_emb_w, hidden_units,
                     layer, activation=tf.sigmoid, att_part=3, reuse=None, name="items"):
    if layer == 0:
        return item_emb_w, tf.summary.merge([])

    fin = item_emb_w
    fin_list = [fin]
    for i in range(layer):
        # print(i)
        fin = tf.sparse_tensor_dense_matmul(support_in_sp, fin)    # n*n  n*d/2
        fin_list.append(fin)

    w0 = tf.layers.dense(fin_list[0], 1, activation=activation, name='{}_att{}_gcn0'.format(name, att_part), reuse=reuse)
    w0 = tf.reduce_mean(w0)

    summary_list = [tf.summary.scalar('w0', w0)]
    fin = w0 * fin_list[0]
    for i in range(layer):
        if att_part == 3:
            weight = tf.reduce_mean(attention_three_part_gcn_layer(fin_list[0], fin_list[i + 1], activation=activation,
                                                                   name='{}_att3_gcn{}'.format(name, i + 1), reuse=reuse))
        elif att_part == 4:
            weight = tf.reduce_mean(attention_four_part_gcn_layer(fin_list[0], fin_list[i + 1], activation=activation,
                                                                  name='{}_att4_gcn{}'.format(name, i + 1), reuse=reuse))
        else:
            raise ValueError("please define the correct param: att_part")

        # weight_list.append(weight)
        fin += fin + weight * fin_list[i + 1]
        weight_summary = tf.summary.scalar('w{}'.format(i + 1), weight)
        summary_list.append(weight_summary)

    # merged = tf.summary.merge(summary_list)

    return fin, summary_list

# def gcn_my_att_layer(support_in_sp, item_emb_w, hidden_units,
#                      layer, activation=tf.sigmoid, att_part=3, reuse=None, name="items"):
#     if layer == 0:
#         return item_emb_w, None
#     fin = item_emb_w
#     print("fin", fin)
#     fin_list = [fin]
#     print("fin_list", fin_list)

#     for i in range(layer):
#         print(i)
#         fin = tf.sparse_tensor_dense_matmul(support_in_sp, fin)    # n*n  n*d/2
#         fin_list.append(fin)

#     w0 = tf.layers.dense(fin_list[0], 1, activation=activation, name='{}_att{}_gcn0'.format(name, att_part), reuse=reuse)
#     w0 = tf.reduce_mean(w0)

#     fin = w0 * fin_list[0]
#     for i in range(layer):
#         if att_part == 3:
#             weight = tf.reduce_mean(attention_three_part_gcn_layer(fin_list[0], fin_list[i + 1], activation=activation,
#                                                                    name='{}_att3_gcn{}'.format(name, i + 1), reuse=reuse))
#         elif att_part == 4:
#             weight = tf.reduce_mean(attention_four_part_gcn_layer(fin_list[0], fin_list[i + 1], activation=activation,
#                                                                   name='{}_att4_gcn{}'.format(name, i + 1), reuse=reuse))
#         else:
#             raise ValueError("please define the correct param: att_part")

#         # fin += fin + weight * fin_list[i + 1]
#         fin +=  weight * fin_list[i + 1]
 

#     return fin, None

# def gcn_my_att_layer(support_in_sp, item_emb_w, hidden_units,
#                      layer, activation=tf.sigmoid, att_part=3, reuse=None, name="items"):
#     if layer == 0:
#         return item_emb_w, None

#     fin = item_emb_w  # x0
#     w0 = tf.layers.dense(item_emb_w, 1, activation=activation, name='{}_att{}_gcn0'.format(name, att_part), reuse=reuse)
#     w0 = tf.reduce_mean(w0)   # a0

#     res = w0 * item_emb_w   # x_out
#     for i in range(layer):
#         fin = tf.sparse_tensor_dense_matmul(support_in_sp, fin)   # x_i
#         # a_i
#         if att_part == 3:
#             weight = tf.reduce_mean(attention_three_part_gcn_layer(item_emb_w, fin, activation=activation,
#                                                                    name='{}_att3_gcn{}'.format(name, i + 1), reuse=reuse))
#         elif att_part == 4:
#             weight = tf.reduce_mean(attention_four_part_gcn_layer(item_emb_w, fin, activation=activation,
#                                                                   name='{}_att4_gcn{}'.format(name, i + 1), reuse=reuse))
#         else:
#             raise ValueError("please define the correct param: att_part")
#         res +=  weight * fin

#     return res, None

def gcn_layer(support_in_sp, item_emb_w, hidden_units, layer, activation=tf.sigmoid):
    fin = item_emb_w

    for i in range(layer):
        fin = tf.sparse_tensor_dense_matmul(support_in_sp, fin)

    return fin, None


def gat_layer(support_in_sp, item_emb_w, hidden_units, layer, activation=tf.sigmoid):
    item_count = 63001
    gat_w = tf.get_variable("gat_w", [item_count, hidden_units // 2])

    fin = item_emb_w

    for i in range(layer):
        fin = tf.sparse_tensor_dense_matmul(support_in_sp, fin)

    return fin, None


# ------------------------------------------------------------------------------------------
# unuse

def age_layer(support_in_sp, item_emb_w, hidden_units, layer, activation=tf.sigmoid):
    fin = item_emb_w

    for i in range(layer):
        xixi1 = tf.sparse_tensor_dense_matmul(support_in_sp, fin)
        fin = 1 / 3 * fin + 2 / 3 * xixi1

    return fin, None


def agc_layer(support_in_sp, item_emb_w, hidden_units, layer, activation=tf.sigmoid):
    fin = item_emb_w

    for i in range(layer):
        xixi1 = tf.sparse_tensor_dense_matmul(support_in_sp, fin)
        fin = 1 / 2 * fin + 1 / 2 * xixi1

    return fin, None
