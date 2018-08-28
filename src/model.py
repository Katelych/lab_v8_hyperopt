import timeit
import operator
import numpy as np
import tensorflow as tf
from dataset import KnowledgeGraph



class RelationPredict:
    def __init__(self, kg: KnowledgeGraph, batch_size, learning_rate):     # ':' is type hint
        self.kg = kg
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        '''ops for training'''
        self.train_nodes_id = tf.placeholder(dtype=tf.int32, shape=[batch_size,])
        self.train_nodes_label = tf.placeholder(dtype=tf.float32, shape=[batch_size, kg.groups_num])
        self.train_nodes_neighbor = tf.placeholder(dtype=tf.float32, shape=[None, kg.groups_num])
        self.train_nodes_neighborID = tf.placeholder(dtype=tf.int32, shape=[None,])
        self.train_nodes_neighborNum = tf.placeholder(dtype=tf.float32, shape=[batch_size,])
        #self.predict_emb = None
        self.train_op = None
        self.loss = None
        self.accuracy = None
        self.epoch_accuracy = None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None
        '''relation matrix'''
        '''
        with tf.variable_scope('relation_matrix'):
            self.relation_matrix = tf.get_variable(name='relation',
                                          shape=[kg.groups_num, kg.groups_num],
                                          dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(mean=0, stddev=1))
            tf.summary.histogram(name='relation matrix', values=self.relation_matrix)
        with tf.variable_scope('bias'):
            self.bias = tf.Variable(tf.zeros([kg.groups_num],tf.float32), name='bias')
            tf.summary.histogram(name='bias', values=self.bias)
             
        #self.relation_matrix = tf.Variable(initial_value=tf.eye(kg.groups_num), name='relation_matrix')
        '''
        index = tf.stack([np.arange(kg.groups_num), np.arange(kg.groups_num)], axis=1)
        init = tf.sparse_to_dense(index, [kg.groups_num, kg.groups_num], 3., -3.)
        self.relation_matrix = tf.Variable(init, 'relation_matrix')
        tf.summary.histogram(name='relation matrix', values=self.relation_matrix)
        self.bias = tf.Variable(tf.zeros([kg.groups_num],tf.float32), name='bias')
        tf.summary.histogram(name='bias', values=self.bias)
        self.build_graph()
        

    def build_graph(self):
        with tf.name_scope('training'):
            predict_emb = self.inference(self.train_nodes_neighbor, self.train_nodes_neighborID, self.train_nodes_neighborNum)
            tf.summary.histogram(name='predict embed', values=predict_emb)  
            self.loss = self.calculate_loss(predict_emb, self.train_nodes_label)
            tf.summary.scalar(name='loss', tensor=self.loss)
            #optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            optimizer = tf.train.AdamOptimizer() #default learning rate
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.relation_matrix = tf.sigmoid(self.relation_matrix)
            self.accuracy = self.evaluation(predict_emb, self.train_nodes_label)
            tf.summary.scalar(name='accuracy', tensor=self.accuracy)
            self.merge = tf.summary.merge_all()
            
          
    def inference(self, neighbors_embed, neighbors_ids, neighbors_num):
        with tf.name_scope('prepare_data'):
            nei_emb = tf.transpose(tf.cast(neighbors_embed, tf.float32)) #2-D, 39*all_nei 
        with tf.name_scope('predict_batch_nodes_embedding'):
            predict_embed = tf.segment_sum(tf.transpose(tf.matmul(self.relation_matrix,nei_emb)), neighbors_ids)
            predict_embed = predict_embed / tf.expand_dims(neighbors_num, 1) + self.bias
        return predict_embed


    def calculate_loss(self, predict_emb, real_emb):
        with tf.name_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=real_emb, logits=predict_emb)
        return tf.reduce_mean(loss)
    
    def evaluation(self, predict_emb, real_emb):
        with tf.name_scope('evaluation'):
            #correct = tf.nn.in_top_k(predict_emb, tf.argmax(np.array(real_emb),1),1) # group(label) start at 1
            correct = tf.nn.in_top_k(predict_emb, tf.argmax(real_emb,1),1) # group(label) start at 1
        return tf.reduce_sum(tf.cast(correct, tf.int32)) / self.batch_size


    def launch_training(self, session, summary_writer):
        print('------------------Start training----------------')
        print('batch_size: {} learning_rate: {}'.format(self.batch_size, self.learning_rate))
        start = timeit.default_timer()
        batch_num = self.kg.training_num // self.batch_size
        epoch_loss = 0
        used_nodes_num = 0
        epoch_accuracy = 0
        for batch in range(batch_num):
            curr_node_batch, curr_label_batch, nei_embed_batch, nei_ids, nei_num = self.kg.next_batch(self.batch_size)
            #nei_embed_batch = np.float32(nei_embed_batch)
            #print(np.array(nei_embed_batch).dtype)  #float64
            batch_accuracy, batch_loss, _, summary = session.run(fetches=[self.accuracy, self.loss, self.train_op, self.merge],
                                                         feed_dict={self.train_nodes_id: curr_node_batch,
                                                                    self.train_nodes_label: curr_label_batch,
                                                                    self.train_nodes_neighbor: nei_embed_batch,
                                                                    self.train_nodes_neighborID: nei_ids,
                                                                    self.train_nodes_neighborNum: nei_num})
            global_step = self.global_step.eval(session=session)
            summary_writer.add_summary(summary, global_step=global_step)
            epoch_loss += batch_loss
            used_nodes_num += len(curr_node_batch)
            epoch_accuracy += batch_accuracy
            print('[epoch: {} batch: {} global_step: {}] #node: {}/{} avg_loss: {:.6f} batch_accuracy: {}'.format(self.kg.epochs_completed, batch, global_step, used_nodes_num, self.kg.training_num, batch_loss / len(curr_node_batch), batch_accuracy), end='\r')
        self.epoch_accuracy = epoch_accuracy / batch_num
        print()
        print('epoch loss: {:.3f}'.format(epoch_loss))
        print('epoch accuracy: {}'.format(self.epoch_accuracy))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----------------Finish training-----------------')

    
    def check(self, session):
        print('-----------Check----------')
        rela = self.relation_matrix.eval(session=session)
        bias = self.bias.eval(session=session)
        print('relation matrix:')
        print(rela)
        print('bias:')
        print(bias)
        print('--------Check Finished----')
        
if __name__ == '__main__':
    kg = KnowledgeGraph(data_dir='../data/')
    kg.training_num = 4
    model = RelationPredict(kg=kg, batch_size=4, learning_rate=0.01)    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(logdir='../summ', graph=sess.graph)
    model.launch_training(session=sess, summary_writer=summary_writer)
    model.check(session=sess)
    sess.close()
