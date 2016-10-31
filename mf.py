import tensorflow as tf
import random 
import numpy as np

max_iter=10
num_users=7374
num_items=105113
rank=20
lda=1e-5
lr=1e-7
batch_size=20

with tf.name_scope("input") as scope:
    user_indices = tf.placeholder(tf.int32, shape=[None])
    item_indices = tf.placeholder(tf.int32, shape=[None])
    user_indices_val = tf.placeholder(tf.int32, shape=[None])
    item_indices_val = tf.placeholder(tf.int32, shape=[None])
    rating_values = tf.placeholder(tf.float32, shape=[None])
    rating_values_val = tf.placeholder(tf.float32, shape=[None])
    mean_rating = tf.placeholder(tf.float32, shape=[])

W = tf.Variable(tf.truncated_normal([num_users, rank], stddev=0.2, mean=0), name="users")
H = tf.Variable(tf.truncated_normal([num_items, rank], stddev=0.2, mean=0), name="items")
W_plus_bias = tf.concat(1, [W, tf.ones((num_users,1), dtype=tf.float32, name="item_bias_ones")])
H_plus_bias = tf.concat(1, [H, tf.ones((num_items,1), name="user_bias_ones", dtype=tf.float32)])
result_values = tf.reduce_sum(tf.mul(tf.gather(W, user_indices), tf.gather(H, item_indices)), 1, name="extract_training_ratings")
result_values_val = tf.reduce_sum(tf.mul(tf.gather(W, user_indices_val), tf.gather(H, item_indices_val)), 1, name="extract_training_ratings")
diff_op = tf.sub(tf.add(result_values, mean_rating, name="add_mean"), rating_values, name="raw_training_error")
diff_op_val = tf.sub(tf.add(result_values_val, mean_rating, name="add_mean_val"), rating_values_val, name="raw_validation_error")

with tf.name_scope("training_cost") as scope:
    base_cost = tf.reduce_sum(tf.square(diff_op, name="squared_difference"), name="sum_squared_error")
    regularizer = tf.mul(tf.add(tf.reduce_sum(tf.square(W)), tf.reduce_sum(tf.square(H))), lda, name="regularize")
    cost = tf.div(tf.add(base_cost, regularizer), tf.to_float(tf.shape(rating_values)[0]), name="average_error")

with tf.name_scope("validation_cost") as scope:
    cost_val = tf.div(tf.reduce_sum(tf.square(diff_op_val, name="squared_difference_val"), name="sum_squared_error_val"), tf.to_float(tf.shape(rating_values_val)[0]), name="average_error")
    rms_val = tf.sqrt(cost_val)

# Use an exponentially decaying learning rate.
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)

with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(cost, global_step=global_step)
with tf.device("/gpu:0"):
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

for i in range(max_iter):
    lines = open("../mf_dataset/ciao/regular_rating_matrix.txt").readlines()
    data = [tuple(line.strip().split("\t", 3)) for line in lines]
    data = [(int(row[0]), int(row[1]), float(row[2])) for row in data]
    random.shuffle(data)
    split_point = int(len(data) * 0.9)
    train_data = data[:split_point]
    val_data = data[split_point:]
    total_loss = 0
    for j in range(len(data)/batch_size):
        U = np.array([u for u, _, _ in train_data[j*batch_size:(j+1)*batch_size]], dtype=np.int32)
        I = np.array([i for _, i, _ in train_data[j*batch_size:(j+1)*batch_size]], dtype=np.int32)
        R = np.array([r for _, _, r in train_data[j*batch_size:(j+1)*batch_size]], dtype=np.float32)
        with tf.device("/gpu:0"):
            loss,  _ = sess.run([cost, train_step], feed_dict={user_indices:U, item_indices:I, rating_values:R, mean_rating:3.0})
            print loss
            total_loss += loss
    print "avg train loss=%f" %(total_loss/(len(data)/batch_size))
    val_U = np.array([u for u, _, _ in val_data], dtype=np.int32)
    val_I = np.array([i for _, i, _ in val_data], dtype=np.int32)
    val_R = np.array([r for _, _, r in val_data], dtype=np.float32)
    with tf.device("/gpu:0"):
        loss, _ = sess.run([rms], feed_dict={user_indices_val:val_U, item_indices_val:val_I, rating_values_val:val_R, mean_rating:3.0})
