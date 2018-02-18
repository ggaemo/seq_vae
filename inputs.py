import tensorflow as tf
import numpy as np
import reader


raw_data = reader.ptb_raw_data('simple-examples/data')

train_data, valid_data, test_data, vocab_size, word_to_id = raw_data

id_to_word = {v: k for k, v in word_to_id.items()}

id_to_word[0] = '<pad>'


def preprocess(data_type):

    if data_type == 'train':
        data = train_data
    elif data_type == 'valid':
        data = valid_data
    elif data_type == 'test':
        data = test_data

    partitioned = '-'.join(map(str, data)).split('-3-')

    partitioned = [x + '-3' for x in partitioned]

    partitioned[-1] = partitioned[-1][:-2]

    partitioned_data = [[int(x) for x in x.split('-')] for x in partitioned]

    partitioned_data.sort(key=len)

    max_len = max(len(x) for x in partitioned_data)

    arr = np.zeros((len(partitioned_data), max_len), dtype=int)

    for idx, val in enumerate(partitioned_data):
        sub_len = len(val)
        arr[idx, :sub_len] = val

    sequence_length = [len(x) for x in partitioned_data]
    return arr, sequence_length, max_len

train_arr, trn_seq_len, trn_max_len = preprocess('train')
test_arr, test_seq_len, test_max_len = preprocess('valid')
valid_arr, valid_seq_len, valid_max_len = preprocess('test')


def inputs(data_type, batch_size):

    if data_type == 'train':
        data = train_arr
        seq_len = trn_seq_len
    elif data_type == 'valid':
        data = valid_arr
        seq_len = valid_seq_len
    elif data_type == 'test':
        data = test_arr
        seq_len = test_seq_len

    ds = tf.data.Dataset.from_tensor_slices((data, seq_len))
    ds = ds.batch(batch_size)
    ds = ds.shuffle(buffer_size=5000)

    iterator = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)
    next_element = iterator.get_next()
    init_op = iterator.make_initializer(ds)

    return next_element, init_op


