'''
A Linear Regression implementation example using TensorFlow library.
'''

import tensorflow as tf


def inference(X,W,b):

    # compute inference model over data X and return the result
    return tf.matmul(X, W) + b

def loss(X, Y,W,b):

    # compute loss over training data X and expected outputs Y
    Y_predicted = inference(X,W,b)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))

def inputs():

    # read/generate input training data X and expected outputs Y
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content)

def train(total_loss):

    # train / adjust model parameters according to computed total loss
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y,W,b):

    # evaluate the resulting trained model
    print(sess.run(inference([[80., 25.]],W,b)))
    print(sess.run(inference([[65., 25.]],W,b)))



def main():

    #initialize variables/model parameters
    W = tf.Variable(tf.zeros([2, 1]), name="weights")
    b = tf.Variable(0., name="bias")

    # model definition code…
    sess=tf.Session()

    # Create a saver.
    saver = tf.train.Saver()

    X, Y = inputs()
    total_loss = loss(X, Y,W,b)
    train_op = train(total_loss)

    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and learning purposes, see how the loss gets decremented thru training steps
        if step % 10 == 0:
            print("loss: ", sess.run([total_loss]))
            saver.save(sess, 'my-model', global_step=step)



    evaluate(sess, X, Y,W,b)
    saver.save(sess, 'my-model', global_step=training_steps)

    sess.close()


if __name__ == '__main__':
    main()