import inputs
import tensorflow as tf
import numpy as np
import model
import time
batch_size = 96
embedding_size = 30
max_len = 78
hidden_size = 30
learning_rate=  1e-4
grad_clip = 4
vocabulary_size = inputs.vocab_size + 1
id_to_word = inputs.id_to_word

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

inputs_pl = tf.placeholder(tf.int32, shape=[None, None])
inputs_seq_len_pl = tf.placeholder(tf.int32, shape=[None])


with tf.name_scope('train'):
    train_inputs, train_init_op = inputs.inputs('train', batch_size)
    with tf.variable_scope('Model'):
        trn_model = model.Sequential_VAE(
            inputs_pl, inputs_seq_len_pl, vocabulary_size, embedding_size,
            hidden_size, learning_rate, grad_clip)

        # trn_model = model.Sequential_VAE(
        #     inputs_pl, inputs_seq_len_pl, vocabulary_size, embedding_size,
        #          hidden_size, learning_rate, grad_clip)

# with tf.name_scope('Test'):
#     test_inputs, test_init_op = inputs.inputs('test', batch_size)
#     with tf.variable_scope('Model'):
#         test_model = model.Sequential_VAE(
#             inputs_pl, inputs_seq_len_pl, vocabulary_size, embedding_size,
#                  hidden_size, learning_rate, grad_clip)

count = 0
epoch = 0

sess.run(tf.global_variables_initializer())
sess.run(train_init_op)
start = time.time()
while (True):
    try:
        batch, seq_len = sess.run(train_inputs)
    except tf.errors.OutOfRangeError:
        print("epoch took {:.3f} minutes".format((time.time() - start) / 60))
        start = time.time()
        sess.run(train_init_op)
        epoch += 1
        count = 0
    feed_dict = {inputs_pl: batch, inputs_seq_len_pl: seq_len}
    sess.run([trn_model.train_op], feed_dict=feed_dict)
    count += 1
    if count % 1 == 0:
        s_id, loss, g_norm, xent, logits = sess.run([trn_model.sample_id, trn_model.loss,
                                trn_model.global_grad_norm, trn_model.xent_masked,
                                             trn_model.logits],
                              feed_dict=feed_dict)
        print('epoch ', epoch, 'count ', count, 'loss', loss, 'grad_norm', g_norm)
        print('predic : ', [id_to_word[x] for x in s_id[0][:seq_len[0] +5]])
        print('answer : ', [id_to_word[x] for x in batch[0][:seq_len[0] + 5]])





