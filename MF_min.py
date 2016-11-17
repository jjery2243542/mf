import tensorflow as tf
import random 
import numpy as np
from math import sqrt

lines = open("./ciao/regular_rating_matrix.txt").readlines()
data = [tuple(line.strip().split("\t", 3)) for line in lines]
data = [(int(row[0]), int(row[1]), float(row[2])) for row in data]
max_u = [e[0] for e in data]
num_users = max(max_u)+1
print num_users

max_i = [e[1] for e in data]
num_items = max(max_i)+1
print num_items

social = [[] for e in range(num_users)]

with open("./ciao/regular_social_matrix.txt") as f:
    for line in f:
        u, f = map(int, line.split('\t',2))
        social[u].append(f)

#pre-process
user_item_set = [set() for i in range(num_users)]
user_rating = [{} for i in range(num_users)]
for user,item,rate in data:
    user_item_set[user].add(item)
    user_rating[user][item] = rate

#print user_item_set[0].intersection(user_item_set[63])

#calculate sim

sim_t = [{} for e in range(num_users)]

for user in range(num_users):
    for friend in social[user]:
        same_set = user_item_set[user].intersection(user_item_set[friend])
        R_ij_fj = 0
        R_ij_square = 0
        R_fj_square = 0
        for item in same_set:
            R_ij_fj += user_rating[user][item] * user_rating[friend][item]
            R_ij_square += user_rating[user][item]**2
            R_fj_square += user_rating[friend][item]**2

        if len(same_set) != 0: 
            sim_t[user][friend] = R_ij_fj / (sqrt(R_ij_square) * sqrt(R_fj_square))
        else:
            sim_t[user][friend] = 0

print "done calcuate sim"

random.shuffle(data)
split_point = int(len(data) * 0.9)
train_data = data[:split_point]
val_data = data[split_point:]

max_iter=10
rank=20
lda=1e-4
lr=1e-1
batch_size=20

with tf.name_scope("input") as scope:
    user_indices = tf.placeholder(tf.int32, shape=[None])
    item_indices = tf.placeholder(tf.int32, shape=[None])
    user_indices_val = tf.placeholder(tf.int32, shape=[None])
    item_indices_val = tf.placeholder(tf.int32, shape=[None])
    rating_values = tf.placeholder(tf.float32, shape=[None])
    rating_values_val = tf.placeholder(tf.float32, shape=[None])
    mean_rating = tf.placeholder(tf.float32, shape=[])
    sim = tf.placeholder(tf.float32, shape=[None])
    user_i_for_sim = tf.placeholder(tf.int32, shape=[None])
    user_f_for_sim = tf.placeholder(tf.int32, shape=[None])

W = tf.Variable(tf.truncated_normal([num_users, rank], stddev=0.2, mean=0), name="users")
H = tf.Variable(tf.truncated_normal([num_items, rank], stddev=0.2, mean=0), name="items")
#W_plus_bias = tf.concat(1, [W, tf.ones((num_users,1), dtype=tf.float32, name="item_bias_ones")])
#H_plus_bias = tf.concat(1, [H, tf.ones((num_items,1), name="user_bias_ones", dtype=tf.float32)])
result_values = tf.reduce_sum(tf.mul(tf.gather(W, user_indices), tf.gather(H, item_indices)), 1, name="extract_training_ratings")
result_values_val = tf.reduce_sum(tf.mul(tf.gather(W, user_indices_val), tf.gather(H, item_indices_val)), 1, name="extract_training_ratings")
diff_op = tf.sub(tf.add(result_values, mean_rating, name="add_mean"), rating_values, name="raw_training_error")
diff_op_val = tf.sub(tf.add(result_values_val, mean_rating, name="add_mean_val"), rating_values_val, name="raw_validation_error")


with tf.name_scope("training_cost") as scope:
    base_cost = tf.reduce_sum(tf.square(diff_op, name="squared_difference"), name="sum_squared_error")
    regularizer = tf.mul(tf.add(tf.reduce_sum(tf.square(W)), tf.reduce_sum(tf.square(H))), lda, name="regularize")
    s_reg = tf.mul(tf.reduce_sum(tf.mul(tf.reduce_sum(tf.square(tf.sub(tf.gather(W,user_i_for_sim),tf.gather(W,user_f_for_sim))),1,keep_dims=True), sim)),0.05)
    cost = tf.div(tf.add_n([base_cost, s_reg, regularizer]), tf.to_float(tf.shape(rating_values)[0]), name="average_error")

with tf.name_scope("validation_cost") as scope:
    cost_val = tf.div(tf.reduce_sum(tf.square(diff_op_val, name="squared_difference_val"), name="sum_squared_error_val"), tf.to_float(tf.shape(rating_values_val)[0]), name="average_error")
    rms_val = tf.sqrt(cost_val)

# Use an exponentially decaying learning rate.
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)

with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(cost, global_step=global_step)

#print 'before'
sess = tf.Session()
sess.run(tf.initialize_all_variables())
#print 'after'

for i in range(max_iter):
    
    total_loss = 0
    for j in range(int(len(train_data)/batch_size)):
        
        U = np.array([u for u, _, _ in train_data[j*batch_size:(j+1)*batch_size]], dtype=np.int32)
        I = np.array([i for _, i, _ in train_data[j*batch_size:(j+1)*batch_size]], dtype=np.int32)
        R = np.array([r for _, _, r in train_data[j*batch_size:(j+1)*batch_size]], dtype=np.float32)
        uu=[]
        ff=[]
        ss=[]
        for user in U:
            #print len(U)
            #print len(social[user])
            for friend in set(social[user]):  
                ss.append(sim_t[user][friend])
                uu.append(user)
                ff.append(friend)
        
        #print sess.run(s_reg, feed_dict={user_indices:U, item_indices:I, rating_values:R, mean_rating:3.0, sim:np.array(ss), user_i_for_sim:np.array(uu), user_f_for_sim:np.array(ff)})
        loss,  _ = sess.run([cost, train_step], feed_dict={user_indices:U, item_indices:I, rating_values:R, mean_rating:3.0, sim:np.array(ss), user_i_for_sim:np.array(uu), user_f_for_sim:np.array(ff)})
        total_loss += loss
        #print 'loss = ',loss

    print "avg_train_loss=%f" %(total_loss/(len(train_data)/batch_size))
    val_U = np.array([u for u, _, _ in val_data], dtype=np.int32)
    val_I = np.array([i for _, i, _ in val_data], dtype=np.int32)
    val_R = np.array([r for _, _, r in val_data], dtype=np.float32)
    
    rms = sess.run([rms_val], feed_dict={user_indices_val:val_U, item_indices_val:val_I, rating_values_val:val_R, mean_rating:3.0})
    print "val_rms=%f" %(rms[0])