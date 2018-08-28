from dataset import KnowledgeGraph
from model import RelationPredict
import tensorflow as tf
import argparse


def main():
    parser = argparse.ArgumentParser(description='predict relation matrix in social network')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--summary_dir', type=str, default='../summary/')
    parser.add_argument('--epoch_num', type=int, default=500)
    parser.add_argument('--eval_freq', type=int, default=10)
    args = parser.parse_args()
    print(args)
    #kg = KnowledgeGraph(data_dir=args.data_dir)
    #kge_model = RelationPredict(kg=kg, batch_size=args.batch_size, learning_rate=args.learning_rate)
    gpu_config = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_config)
    with tf.Session(config=sess_config) as sess:
        '''
        print('----------Initializing of graph----------')
        tf.global_variables_initializer().run()
        print('----------Initialization accomplished-------')
        print('Initial relation_matrix:')
        print(kge_model.relation_matrix.eval(session=sess))
        print('Initial bias:')
        print(kge_model.bias.eval(session=sess))
        summary_writer = tf.summary.FileWriter(logdir=args.summary_dir, graph=sess.graph)
        for epoch in range(args.epoch_num):
            print('=' * 30 + '[EPOCH {}]'.format(epoch) + '=' * 30)
            kge_model.launch_training(session=sess, summary_writer=summary_writer)
            kge_model.check(session=sess)
        print('---------------------ENDING-----------------------')
        '''
        
        #kg = KnowledgeGraph(data_dir=args.data_dir)
        def accuracy(params,session=sess):
            kg = KnowledgeGraph(data_dir=args.data_dir)
            kge_model = RelationPredict(kg=kg, batch_size=params['batch_size'], learning_rate=params['learning_rate'])
            tf.global_variables_initializer().run()
            summary_writer = tf.summary.FileWriter(logdir=args.summary_dir, graph=session.graph)
            for epoch in range(args.epoch_num):
                kge_model.launch_training(session=session, summary_writer=summary_writer)
            return -kge_model.epoch_accuracy 

        from hyperopt import fmin, tpe, hp, partial
        space = {'batch_size': hp.choice('batch_size', range(8,200)),
                 'learning_rate': hp.uniform('learning_rate', 0.0001, 10)}
        algo = partial(tpe.suggest, n_startup_jobs=10)
        best = fmin(fn=accuracy, space=space, algo=algo, max_evals=500)
        print(best)
        print(-accuracy(best))
      

if __name__ == '__main__':
    main()
