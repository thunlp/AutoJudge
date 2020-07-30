from .Models import *
from .DataReader import *
import time
import random
import logging
import argparse
import datetime
import os
import pickle
from functools import reduce
from . import config


def evaluate(g, test_set, batch_size, sess, test_writer, file="Evaluation.txt", step=0):
    t_p, t_n, f_p, f_n = 0, 0, 0, 0
    logging.info("start evaluation")
    for case, plea, law, label, l_c, l_p, l_l in test_set:
        feed_dict = {g['case']: case, g['plea']: plea,
                     g['Label']: label, g['Law']: law,
                     g["dop"]: 1,
                     g['case_len']: l_c,
                     g['plea_len']: l_p,
                     g['Law_len']: l_l}
        training_loss_, predict, summary = sess.run([g['total_loss'],
                                                     g['preds'], g["summary"]],
                                                    feed_dict)
        for i in range(batch_size):
            if label[i][0]:
                if predict[i]:
                    f_p += 1
                else:
                    t_n += 1
            else:
                if predict[i]:
                    t_p += 1
                else:
                    f_n += 1
    logging.info("Evaluation Finished.")

    with open(file, "w") as f:
        if step:
            f.write("steps: %d\n" % step)
        f.write("true positive : %d\n" % t_p)
        f.write("true negative : %d\n" % t_n)
        f.write("false positive : %d\n" % f_p)
        f.write("false negative : %d\n" % f_n)
        if t_p != 0:
            f.write("pos acc:%.3f recall:%.3f" % (t_p / (t_p + f_p), t_p / (t_p + f_n)))
        else:
            f.write("pos acc:%.3f recall:%.3f" % (0, 0))
        f.write("\n aggregate acc: %.3f" % ((t_p + t_n) / (t_p + t_n + f_p + f_n)))
    logging.info("Evaluation Finished. Saved to %s" % file)
    test_writer.add_summary(summary, step)


def evaluate_test(g, test_set, batch_size, sess, test_writer, id2tok_word, file="Evaluation.txt", step=0):
    t_p, t_n, f_p, f_n = 0, 0, 0, 0
    logging.info("start evaluation")
    count = 0
    for case, plea, law, label, l_c, l_p, l_l in test_set:
        print("Epoch: %d" % count)
        count += 1
        feed_dict = {g['case']: case, g['plea']: plea,
                     g['Label']: label, g['Law']: law,
                     g["dop"]: 1,
                     g['case_len']: l_c,
                     g['plea_len']: l_p,
                     g['Law_len']: l_l}
        if config.attention:
            attentions_CL, attentions_CP = sess.run([
                g["attentions_CL"],
                g["attentions_CP"]],
                feed_dict)
        else:
            training_loss_, predict, CNN_features = sess.run([g['total_loss'], g['preds'], g["CNN_features"]],
                                                             feed_dict)

        if config.attention:
            for i in range(batch_size):
                case_sum = [case[i], sen_id2tok(case[i], id2tok_word), sen_id2tok(plea[i], id2tok_word),
                            sen_id2tok(law[i], id2tok_word), attentions_CL[i], attentions_CP[i]]
                num = hash(case_sum[1] + case_sum[2])
                with open(outputs + str(num) + ".bin", "wb")as f:
                    pickle.dump(case_sum, f)
        else:
            for i in range(batch_size):
                if label[i][0]:
                    if predict[i]:
                        f_p += 1
                    else:
                        t_n += 1
                else:
                    if predict[i]:
                        t_p += 1
                    else:
                        f_n += 1

            with open("./model/Acc_len.txt", 'w')as f:
                print(label[i][0] == predict[i], l_l[i] + l_p[i], file=f)
            for i in range(batch_size):
                case_sum = [case[i], sen_id2tok(case[i], id2tok_word), sen_id2tok(plea[i], id2tok_word),
                            sen_id2tok(law[i], id2tok_word), label[i], predict[i], CNN_features[i]]
                num = hash(case_sum[1] + case_sum[2])
                with open(outputs + str(num) + "_result.bin", "wb")as f:
                    pickle.dump(case_sum, f)
    if not config.attention:
        with open(file, "w") as f:
            if step:
                f.write("steps: %d\n" % step)
            f.write("true positive : %d\n" % t_p)
            f.write("true negative : %d\n" % t_n)
            f.write("false positive : %d\n" % f_p)
            f.write("false negative : %d\n" % f_n)
            if t_p != 0:
                f.write("pos acc:%.3f recall:%.3f" % (t_p / (t_p + f_p), t_p / (t_p + f_n)))
            else:
                f.write("pos acc:%.3f recall:%.3f" % (0, 0))
            f.write("\n aggregate acc: %.3f" % ((t_p + t_n) / (t_p + t_n + f_p + f_n)))

        logging.info("Evaluation Finished. Saved to %s" % file)

    logging.info("Evaluation Finished.")


def train_network(g, num_epochs, raw_data,  # raw data from
                  Law, Law_len, tok2id, log_file_path,
                  batch_size=32,
                  verbose=True, save=False,
                  eva=True, eva_often=False,
                  restore=config.restore):  # Law [batch*len]
    global dropout_keep_probability
    tf.set_random_seed(random.randint(1000, 3000))
    global num  # marker
    with tf.Session() as sess:
        if len(restore) == 0:
            sess.run(tf.global_variables_initializer())
            logging.info("Initialized")
        else:
            g['saver'].restore(sess, restore)
            logging.info('restored')
        train_writer = tf.summary.FileWriter(log_file_path + "Train", sess.graph)
        test_writer = tf.summary.FileWriter(log_file_path + "Dev", sess.graph)
        logging.info("Tensorboard ready.")
        training_losses = []
        gen = get_epoch(raw_data=raw_data, batch_size=config.batch_size, epoch=num_epochs, tok2id=tok2id,
                        length_trial=config.length_trial, all_petition=config.all_petition,
                        random_input=config.random_input, drop_=config.drop_, save_folder_name=config.save_folder_name, restore=restore, Law=Law,
                        FLAGS=config.FLAGS)
        logging.info("Data Prepared")
        total_case_num = next(gen)
        test_set = next(gen)
        if eva:  # only execute when --eva is given
            evaluate(g, data_feeder(raw_data=test_set[0], batch_size=test_set[1], tok2id=test_set[2], Law=Law,
                                    FLAGS=config.FLAGS), batch_size, sess, test_writer,
                     log_file_path + "before_training.txt", 0)
        logging.info("Evaluation before training finished.")
        # t=time.time()
        for idx, epoch in enumerate(gen):
            training_loss = 0
            steps = 0
            for case, plea, law, label, l_c, l_p, l_l in epoch:
                steps += 1
                feed_dict = {g['case']: case, g['plea']: plea,
                             g['Label']: label, g['Law']: law,
                             g["dop"]: dropout_keep_probability,
                             g['case_len']: l_c,
                             g['plea_len']: l_p,
                             g['Law_len']: l_l}
                training_loss_, acc_, _, _, summary = sess.run([g['total_loss'], g["acc"],
                                                                g['train_step'], g["Logits"], g["summary"]],
                                                               feed_dict)
                train_writer.add_summary(summary, steps + idx * (138000 // batch_size))
                training_loss += float(training_loss_)


                if verbose and steps % 100 == 0:
                    s = "loss: %.3f after %d steps" % (training_loss / 100, steps)
                    logging.info(s)
                    # f.write(s)
                    training_losses.append(training_loss / 100)
                    training_loss = 0
                if steps % config.eva_step == 0 and eva_often:  # g,test_set,batch_size,sess,mark,file="EvaluationResult.txt"
                    evaluate(g, data_feeder(raw_data=test_set[0], batch_size=test_set[1], tok2id=test_set[2], Law=Law,
                                            FLAGS=config.FLAGS), batch_size, sess, test_writer,
                             log_file_path + "%d_%d.txt" % (idx, steps), step=steps)
                    g['saver'].save(sess, save + str(steps) + ".cntk")
            logging.info("epoch %d finished. Ready to do evaluation" % idx)
            evaluate(g, data_feeder(raw_data=test_set[0], batch_size=test_set[1], tok2id=test_set[2], Law=Law,
                                    FLAGS=config.FLAGS), batch_size, sess, test_writer, log_file_path + "%d.txt" % (idx),
                     step=steps)
            logging.info("Eva finished")

            if isinstance(save, str):
                g['saver'].save(sess, save + config.save_folder_name + "_" + str(idx))
    # f.close()
    return training_losses


def detailed_test(g, num_epochs, raw_data,  # raw data from
                  Law, Law_len, tok2id, log_file_path, id2tok_word,
                  batch_size=32,
                  verbose=True, save=False,
                  eva=True, eva_often=False,
                  restore=config.restore):  # Law [batch*len]
    global dropout_keep_probability
    tf.set_random_seed(random.randint(1000, 3000))
    global num  # marker

    with tf.Session() as sess:
        if len(restore) == 0:
            sess.run(tf.global_variables_initializer())
            logging.info("Initialized")
        else:
            g['saver'].restore(sess, restore)
            logging.info('restored')
        train_writer = tf.summary.FileWriter(log_file_path + "Train", sess.graph)
        test_writer = tf.summary.FileWriter(log_file_path + "Dev", sess.graph)
        logging.info("Tensorboard ready.")
        gen = get_epoch(raw_data=raw_data, batch_size=batch_size, epoch=num_epochs, tok2id=tok2id,
                        length_trial=config.length_trial, all_petition=config.all_petition,
                        random_input=config.random_input, drop_=config.drop_, save_folder_name=config.save_folder_name, restore=restore, Law=Law,
                        FLAGS=config.FLAGS)
        logging.info("Data Prepared")
        total_case_num = next(gen)
        test_set = next(gen)
        del gen  # only execute when --eva is given
        evaluate_test(g, data_feeder(raw_data=test_set[0], batch_size=test_set[1], tok2id=test_set[2], Law=Law,
                                     FLAGS=config.FLAGS), batch_size, sess, test_writer, id2tok_word,
                      log_file_path + "before_training.txt", 0)
        logging.info("Evaluation before training finished.")


def log_params():
    for param in dir(config):
        logging.info('%s = %s' % (param, getattr(config, param)))


if __name__ == "__main__":
    if not config.test_mode:
        log_file_path = "./model/log/" + config.save_folder_name + "/"
    else:
        log_file_path = "./model/log/test/" + config.save_folder_name + "/"
        outputs = log_file_path + "output/"
        if not os.path.isdir(outputs):
            os.makedirs(outputs)
    if not os.path.isdir(log_file_path):
        os.makedirs(log_file_path)
    if not os.path.isdir(log_file_path + 'save/'):
        os.makedirs(log_file_path + 'save/')

    num = str(len(os.listdir(log_file_path)))
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_file_path + 'running-%s.log' % num,
                        filemode='w')

    r = raw_data_import(path=config.path,
                        filenames=config.filenames,
                        segmentation_tool=config.segmentation_tool,
                        Law_text=config.Law_text,
                        Text_path=config.Text_path,
                        all_petition=config.all_petition)
    logging.info("Data preprocessed.")

    corpus_path = next(r)
    _, _ = next(r)
    Law_len = 0
    logging.info("Law text finished.")
    tok2id_word, id2tok_word = load_dict(config.dict_path_word, corpus_path, max_vocab=config.vocab_size, pre_wv=config.pre_wv)
    logging.info("Dict built.")

    Law = None  # tokenized

    GRAPH = [AutoJudge]

    graph = GRAPH[config.Graph_num](config.FLAGS)

    print("Graph built")
    log_params()
    logging.info("Graph built")
    if not config.test_mode:
        training_losses = train_network(g=graph, num_epochs=config.epoch,
                                        raw_data=r, Law=Law,
                                        Law_len=None,
                                        tok2id=tok2id_word, log_file_path=log_file_path, batch_size=config.batch_size,
                                        verbose=True, save=log_file_path + 'save/', eva=config.eva_before, eva_often=config.eva_often,
                                        restore=config.restore)
        logging.info("Training finished.")

        with open(log_file_path + "save_loss.dat", "wb") as f:
            pickle.dump(training_losses, f)
        logging.info("losses data saved.")
        print("saved")
    else:
        print("\n\n\n\nTest mode: \n\n\n\n")
        detailed_test(g=graph, num_epochs=config.epoch,
                      raw_data=r, Law=Law,
                      Law_len=None,
                      tok2id=tok2id_word, log_file_path=log_file_path, batch_size=64,
                      verbose=True, save=log_file_path + 'save/', eva=config.eva_before, eva_often=config.eva_often, restore=config.restore,
                      id2tok_word=id2tok_word)
