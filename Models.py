import tensorflow as tf
from Units import *
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import variance_scaling_initializer as VSI
import numpy as np
from yellowfin import *

opt = {"Adam": tf.train.AdamOptimizer, "YF": YFOptimizer}


def AutoJudge(flags):
    tf.reset_default_graph()

    batch_size = flags.batch_size  # Batch-size
    embed_dim_word = flags.embed_dim_word  # Dimension of Embedding of words
    state_size = flags.state_size  # Dimension of hidden states(bi-RNN)
    vocab_size = flags.vocab_size  # size of Vocabulary
    GRU_stack_num = flags.GRU_stack_num
    encode_GRU_stack_num = flags.encode_GRU_stack_num
    learning_rate = flags.learning_rate
    pre_trained_WV = flags.pre_wv  # "": joint learning; str: the pre_trained wv file
    filter_sizes = flags.filter_sizes
    num_filters = flags.num_filters
    hdim = 2 * state_size
    trans = tf.constant([[1, 0], [0, 1]], dtype=tf.int32, shape=[2, 2], name="trans")
    test_mode = flags.test_mode
    """
    Placeholders for passage and question
    input text should be a series of number tokens
    """
    case = tf.placeholder(tf.int32, shape=[batch_size, None], name='case')  # case description
    plea = tf.placeholder(tf.int32, shape=[batch_size, None], name='plea')  # plea
    Law = tf.placeholder(tf.int32, shape=[batch_size, None], name='law')  # law text
    Label = tf.placeholder(tf.int32, shape=[batch_size, 2], name='judgement')  # batch* (1/0 1/0)
    case_len = tf.placeholder(tf.int32, name="case_len")
    plea_len = tf.placeholder(tf.int32, name="plea_len")
    Law_len = tf.placeholder(tf.int32, name="law_len")
    dropout_keep_probability = tf.placeholder(tf.float32, shape=[], name="drop_prob")  # scalar
    # Embedding

    if len(pre_trained_WV) == 0:
        Embedding = tf.get_variable('Embedding', dtype=tf.float32, shape=[vocab_size, embed_dim_word],
                                    initializer=VSI(mode='FAN_AVG'))
        # E_Char = tf.get_variable('E_Char', dtype=tf.float32, shape=[char_count, embed_dim_char], initializer=xinit())
    else:
        with open("/data/disk1/private/LSB/model/Dict_v_%d.txt" % embed_dim_word, "r")as f:
            Embedding = f.readlines()
            Embedding = np.array(list(map(lambda x: list(map(float, x.strip().split()[3:])), Embedding)) + [
                [0 for _ in range(embed_dim_word)] for _ in range(2)])
            Embedding = tf.constant(Embedding, dtype=tf.float32, name="Embedding_Const")
            # Embedding file should be strings that are in the following format:  id_token,D1,D2,....

    # embedding  batch * time * embedding_dimension
    case_embedding = tf.nn.embedding_lookup(Embedding, case)
    plea_embedding = tf.nn.embedding_lookup(Embedding, plea)
    law_embedding = tf.nn.embedding_lookup(Embedding, Law)

    ##case Encoder
    with tf.variable_scope("case_encoding"):
        cell_case_f = gru_n(state_size, GRU_stack_num)
        cell_case_b = gru_n(state_size, GRU_stack_num)
        case_states, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_case_f,
                                                         cell_bw=cell_case_b,
                                                         inputs=case_embedding,
                                                         initial_state_fw=cell_case_f.zero_state(batch_size,
                                                                                                 tf.float32),
                                                         initial_state_bw=cell_case_b.zero_state(batch_size,
                                                                                                 tf.float32),
                                                         sequence_length=case_len,
                                                         time_major=False, scope="case_encoding")  # batch * time * 2dim

    case_states = tf.concat([case_states[0], case_states[1]], axis=2)

    # plea Encoder
    with tf.variable_scope("plea_encoding"):
        cell_plea_f = gru_n(state_size, GRU_stack_num)
        cell_plea_b = gru_n(state_size, GRU_stack_num)
        plea_states, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_plea_f,
                                                         cell_bw=cell_plea_b,
                                                         inputs=plea_embedding,
                                                         initial_state_fw=cell_plea_f.zero_state(batch_size,
                                                                                                 tf.float32),
                                                         initial_state_bw=cell_plea_b.zero_state(batch_size,
                                                                                                 tf.float32),
                                                         sequence_length=plea_len,
                                                         time_major=False, scope="plea_encoding")  # batch * time * 2dim

    plea_states = tf.concat([plea_states[0], plea_states[1]], axis=2)

    # law Encoder
    with tf.variable_scope("law_encoding"):
        cell_law_f = gru_n(state_size, GRU_stack_num)
        cell_law_b = gru_n(state_size, GRU_stack_num)
        law_states, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_law_f,
                                                        cell_bw=cell_law_b,
                                                        inputs=law_embedding,
                                                        initial_state_fw=cell_law_f.zero_state(batch_size, tf.float32),
                                                        initial_state_bw=cell_law_b.zero_state(batch_size, tf.float32),
                                                        sequence_length=Law_len,
                                                        time_major=False, scope="law_encoding")  # batch * time * 2dim

    law_states = tf.concat([law_states[0], law_states[1]], axis=2)

    # case-aware Law Representation

    with tf.variable_scope('CL-matching'):
        cell_CL = Gated_Attention_Cell(num_units=hdim, core=case_states, batch_size=batch_size, scope="GA_CL",
                                       attention_mode=flags.attention)
        CL_states, _ = tf.nn.dynamic_rnn(cell=cell_CL,
                                         inputs=law_states,
                                         initial_state=cell_CL.zero_state(batch_size, tf.float32),
                                         time_major=False,
                                         scope="CL_match")

    """
    Case-aware plea Representation
    """

    with tf.variable_scope('CP-matching'):
        cell_CP = Gated_Attention_Cell(num_units=hdim, core=case_states, batch_size=batch_size, scope="GA_CP",
                                       attention_mode=flags.attention)
        CP_states, _ = tf.nn.dynamic_rnn(cell=cell_CP,
                                         inputs=plea_states,
                                         initial_state=cell_CP.zero_state(batch_size, tf.float32),
                                         time_major=False,
                                         scope="CP_match")  # batch*time*dim
    """
    CNN
    """
    if flags.attention:
        return dict(
            case=case,  # case description     input
            plea=plea,  # plea                 input
            Law=Law,  # law text               input
            Label=Label,  # batch* (1/0 1/0)  input
            case_len=case_len,  # input
            plea_len=plea_len,  # input
            Law_len=Law_len,  # input
            saver=tf.train.Saver(),
            dop=dropout_keep_probability,  # input
            attentions_CL=CL_states,
            attentions_CP=CP_states
        )

    penultimate = conv(inputs=tf.concat([CL_states, CP_states], axis=1),
                       embedding_dimension=hdim,
                       filter_sizes=filter_sizes,
                       num_filters=num_filters,
                       dropout_keep_probability=dropout_keep_probability,
                       batch_size=batch_size) / dropout_keep_probability  # * flags.dropout_keep_probability

    features = tf.summary.histogram("Feature_Layer_Distribution", penultimate)

    """
    Output Layer
    """
    Logits = fully_connected(inputs=penultimate, num_outputs=2, activation_fn=None)  # batch*num_classes
    total_loss = tf.reduce_mean(
        tf.losses.sigmoid_cross_entropy(multi_class_labels=Label, logits=Logits, loss_collection=tf.GraphKeys.LOSSES))
    predictions = tf.argmax(tf.nn.softmax(Logits), axis=1)
    acc = tf.contrib.metrics.accuracy(labels=Label, predictions=tf.nn.embedding_lookup(trans, predictions))
    train_step = opt[flags.optimizer](learning_rate).minimize(total_loss)

    loss = tf.summary.scalar("Loss_Per_Step", total_loss)
    prec = tf.summary.scalar("precision", acc)
    merged = tf.summary.merge_all()
    """
        qlen=q_len,
        plen=p_len,
        Lawlen=Law_len,
    """
    return dict(
        case=case,  # case description     input
        plea=plea,  # plea                 input
        Law=Law,  # law text               input
        Label=Label,  # batch* (1/0 1/0)  input
        case_len=case_len,  # input
        plea_len=plea_len,  # input
        Law_len=Law_len,  # input
        dop=dropout_keep_probability,  # input
        total_loss=total_loss,  # outputs
        train_step=train_step,  # op
        preds=predictions,  # outputs
        saver=tf.train.Saver(),  # op
        Logits=Logits,  # outputs
        acc=acc,
        summary=merged,
        attentions_CL=CL_states,
        attentions_CP=CP_states,
        CNN_features=penultimate
    )
