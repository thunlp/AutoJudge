import tensorflow as tf
import datetime


tf.app.flags.DEFINE_integer("batch_size", 64, "batch size ")
tf.app.flags.DEFINE_integer("embed_dim_word", 100, "the embedding dimension of word")
tf.app.flags.DEFINE_integer("state_size", 100, "size of hidden states in GRU/LSTM")
tf.app.flags.DEFINE_integer("vocab_size", 20004, "size of vocabulary")
tf.app.flags.DEFINE_integer("GRU_stack_num", 2, "num of the layers of GRU (for reading layer)")
tf.app.flags.DEFINE_integer("encode_GRU_stack_num", 1, "num of the layers of GRU (for encoding layer)")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
tf.app.flags.DEFINE_integer("epoch", 5, "number of epoch")
tf.app.flags.DEFINE_string("path", "./raw", "path")
tf.app.flags.DEFINE_string("filenames", "law_seg_refined_data_new_0_30000/law_seg_refined_data_new_30000_60000/law_seg_refined_data_new_60000_90000/law_seg_refined_data_new_90000_120000", "filenames")
tf.app.flags.DEFINE_string("segmentation_tool", "./", "segmentation_tool")
tf.app.flags.DEFINE_string("Law_text", "./raw/divorce.txt", "Law text")
tf.app.flags.DEFINE_string("Text_path", "./model/Text_all.txt", "where to store the whole corpus")
tf.app.flags.DEFINE_boolean("all_petition", True, "all_petition")
tf.app.flags.DEFINE_string("dict_path", "./model/", "where to load/save dictionary")
tf.app.flags.DEFINE_string("dict_name", "Dict_v_100.txt", "dict_name")
tf.app.flags.DEFINE_string("dict_path_word", "./model/Dict.txt", "dict_path_word")
tf.app.flags.DEFINE_string("restore", "", "path to restore trained model")
tf.app.flags.DEFINE_string("pre_wv", "", "path to restore pretrained word vectors")
tf.app.flags.DEFINE_string("save_folder_name", str(datetime.date.today()), "")
tf.app.flags.DEFINE_boolean("eva_before", False, "whether to do evaluation before training")
tf.app.flags.DEFINE_boolean("eva_often", True, "evaluation every 300 steps")
tf.app.flags.DEFINE_boolean("length_trial", False, "reverse")
tf.app.flags.DEFINE_boolean("random_input", True, "random input")
tf.app.flags.DEFINE_float("drop_", 0.01, "drop raw data")
tf.app.flags.DEFINE_string("filter_sizes", "3,4,5", "sizes of filters")
tf.app.flags.DEFINE_integer("num_filters", 100, "the number of filters")
tf.app.flags.DEFINE_integer("eva_step", 250, "eva_step")
tf.app.flags.DEFINE_float("dropout_keep_probability", 0.5, "keep probability for drop out layer")
tf.app.flags.DEFINE_integer("Graph_num", 0, "which graph to use")
tf.app.flags.DEFINE_string("optimizer", "Adam", "opti")
tf.app.flags.DEFINE_boolean("test_mode", False, "test_mode")
tf.app.flags.DEFINE_boolean("attention", False, "attention")


FLAGS = tf.app.flags.FLAGS

batch_size = FLAGS.batch_size
embed_dim_word = FLAGS.embed_dim_word
state_size = FLAGS.state_size
vocab_size = FLAGS.vocab_size
GRU_stack_num = FLAGS.GRU_stack_num
encode_GRU_stack_num = FLAGS.encode_GRU_stack_num
learning_rate = FLAGS.learning_rate
epoch = FLAGS.epoch
path = FLAGS.path
filenames = FLAGS.filenames.split("/")
segmentation_tool = FLAGS.segmentation_tool
Law_text = FLAGS.Law_text
Text_path = FLAGS.Text_path
all_petition = FLAGS.all_petition
dict_path = FLAGS.dict_path
dict_name = FLAGS.dict_name
dict_path_word = FLAGS.dict_path_word
restore = FLAGS.restore
test_mode = FLAGS.test_mode
if len(FLAGS.pre_wv) > 0:
    pre_wv = "Dict_v_%d.txt" % FLAGS.embed_dim_word
    dict_path_word = dict_path + "Dict_v_%d.txt" % FLAGS.embed_dim_word
    print("Dict_path=", dict_path_word)
else:
    pre_wv = ""
save_folder_name = FLAGS.save_folder_name
eva_before = FLAGS.eva_before
eva_often = FLAGS.eva_often
length_trial = FLAGS.length_trial
random_input = FLAGS.random_input
drop_ = FLAGS.drop_
filter_sizes = FLAGS.filter_sizes
num_filters = FLAGS.num_filters
dropout_keep_probability = FLAGS.dropout_keep_probability
eva_step = FLAGS.eva_step
Graph_num = FLAGS.Graph_num
optimizer = FLAGS.optimizer
attention = FLAGS.attention
