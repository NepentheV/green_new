from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False


    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=True,
                 sparse_inputs=True, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):   # N
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GAT(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, sparse_inputs=True,
                 act=tf.nn.elu,  in_drop=0.6, coef_drop=0.6, residual=False, **kwargs):
        super(GAT, self).__init__(**kwargs)

        self.act = act
        self.support = placeholders['support']
        self.residual = residual
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparse_inputs = sparse_inputs
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs  # N*D
        print(x.shape[0])
        # x = tf.SparseTensor(x.indices, x.values, x.dense_shape)
        x = tf.sparse_tensor_to_dense(x)
        print(x)
        seq = tf.reshape(x, (1, x.shape[0], x.shape[1]))
        with tf.name_scope('sp_attn'):
            if self.in_drop != 0.0:
                seq = tf.nn.dropout(seq, 1.0 - self.in_drop)

            seq_fts = tf.layers.conv1d(seq, self.output_dim, 1, use_bias=False)  # 输入1*N*D  输出1*N*D (1*63001*64)   seq_fts=WH

            # simplest self-attention possible
            f_1 = tf.layers.conv1d(seq_fts, 1, 1)  # 输入1*N*D 输出 1*N*1 f_1 = a(Whi)
            f_2 = tf.layers.conv1d(seq_fts, 1, 1)  # 输入1*N*D 输出 1*N*1 f_2 = a(Whj)

            f_1 = tf.reshape(f_1, (x.shape[0], 1))  # N*1
            f_2 = tf.reshape(f_2, (x.shape[0], 1))  # N*1

            # f_1 = tf.cast(f_1, tf.int64)
            # f_2 = tf.cast(f_2, tf.int64)

            f_1 = self.support[0] * f_1  # N*N
            f_2 = self.support[0] * tf.transpose(f_2, [1, 0])  # N*N

            logits = tf.sparse_add(f_1, f_2)  # N*N
            lrelu = tf.SparseTensor(indices=logits.indices,
                                    values=tf.nn.leaky_relu(logits.values),
                                    dense_shape=logits.dense_shape)
            coefs = tf.sparse_softmax(lrelu)  # N*N

            if self.coef_drop != 0.0:
                coefs = tf.SparseTensor(indices=coefs.indices,
                                        values=tf.nn.dropout(coefs.values, 1.0 - self.coef_drop),
                                        dense_shape=coefs.dense_shape)
            if self.in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - self.in_drop)

            # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
            # here we make an assumption that our input is of batch size 1, and reshape appropriately.
            # The method will fail in all other cases!
            coefs = tf.sparse_reshape(coefs, (63001,63001))  # [x.shape[0], x.shape[0]])  # N*N
            seq_fts = tf.squeeze(seq_fts)  # N*D
            vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)  # N*D
            vals = tf.expand_dims(vals, axis=0)
            vals.set_shape([1, x.shape[0], self.output_dim])
            ret = tf.contrib.layers.bias_add(vals)

            # residual connection
            if self.residual:
                if seq.shape[-1] != ret.shape[-1]:
                    ret = ret + tf.layers.conv1d(seq, ret.shape[-1], 1)  # activation
                else:
                    ret = ret + seq

            ret = tf.reshape(ret, (ret.shape[1], ret.shape[2]))
        return self.act(ret)  # activation


class Bilinear(Layer):
    """Bilinear layer."""
    def __init__(self, input_dim, num_features_nonzero=0, dropout=0., sparse_inputs=False,
                 act=tf.nn.sigmoid, bias=False, featureless=False, **kwargs):
        super(Bilinear, self).__init__(**kwargs)

        # if dropout:
        #     self.dropout = placeholders['dropout']
        # else:
        self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        # self.num_features_nonzero = num_features_nonzero

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, input_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([input_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x, y = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
        output = dot(output, y, sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class MeanPooling(Layer):
    def __init__(self, **kwargs):
        super(MeanPooling, self).__init__(**kwargs)

    def _call(self, inputs):
        x = inputs
        x = tf.reduce_mean(x, axis=0)
        return tf.expand_dims(x, axis=-1)
