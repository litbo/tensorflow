import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)
from resnet import resnet18


def preprocess(x, y):
    # [0~1]
    x = tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsz = 256
# [32, 32, 3], [10k, 1]
(x, y), (x_val, y_val) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_val = tf.squeeze(y_val, axis=1)  # 注意维度变换
print(x.shape, y.shape, x_val.shape, y_val.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(batchsz)

test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batchsz)

sample = next(iter(train_db))
print('batch: ', sample[0].shape, sample[1].shape)


def main():
    model = resnet18()
    model.summary()
    model.build(input_shape=(None, 32, 32, 3))
    optimizer = optimizers.Adam(lr=1e-4)

    # 拼接需要训练的参数 [1,2] + [3,4] = [1,2,3,4]
    for epoch in range(50):

        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b,32,32,3] => [b,1,1,512]
                logits = model(x)

                y_onehot = tf.one_hot(y, depth=100)  # [50k, 10]
                # y_val_onehot = tf.one_hot(y_val, depth=100)

                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trianabel_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss: ', float(loss))

        total_num = 0
        total_correct = 0
        for x, y in test_db:
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print('acc: ', acc)


if __name__ == '__main__':
    main()
