import tensorflow as tf
import keras
from keras.src.saving import register_keras_serializable  # 导入Keras的序列化注册器


@register_keras_serializable()  # 注册为可序列化的Keras模型
# 将CNNModel注册为Keras的可序列化模型，方便保存和加载模型
class CNNModel(keras.Model):
    def __init__(self, dtype='float32', **kwargs):
        # 初始化方法中调用父类的初始化方法，然后定义模型的默认数据类型'dtype'
        super(CNNModel, self).__init__()
        self._dtype = dtype

        @property  # 属性方法允许外部访问和设置数据类型
        def dtype(self):
            return self._dtype

        @dtype.setter
        def dtype(self, value):
            self._dtype = value

        # 这个方法用于返回模型的配置信息，方便序列化和反序列化。
        def get_config(self):
            config = super(CNNModel, self).get_config()
            config.updata({
                'dtype': self._dtype
            })
            return config

        # 第一层卷积和ReLU激活函数
        self.conv1 = keras.layers.Conv2D(32, (3, 3), padding='valid', activation='relu')  # , kernel_regularizer=keras.regularizers.l2(0.01)
        # 第二层卷积和ReLU激活函数
        self.conv2 = keras.layers.Conv2D(32, (3, 3), padding='valid', activation='relu')
        # 第一个最大池化层
        self.pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        # 第三层卷积和ReLU激活函数
        self.conv3 = keras.layers.Conv2D(64, (3, 3), padding='valid', activation='relu')
        # 第四层卷积和ReLU激活函数
        self.conv4 = keras.layers.Conv2D(64, (3, 3), padding='valid', activation='relu')
        # 第二个最大池化层
        self.pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        # 第五层卷积和ReLU激活函数
        self.conv5 = keras.layers.Conv2D(128, (3, 3), padding='valid', activation='relu')
        # 第六层卷积和ReLU激活函数
        self.conv6 = keras.layers.Conv2D(128, (3, 3), padding='valid', activation='relu')
        # 第三个最大池化层
        self.pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        # 展开层
        self.flatten = keras.layers.Flatten()
        # Dropout层
        self.dropout = keras.layers.Dropout(rate=0.3)
        # 输出层1到7，每个都是一个全连接层，用于预测每个位置的字符
        self.fc1_1 = keras.layers.Dense(65, activation=None)
        self.fc1_2 = keras.layers.Dense(65, activation=None)
        self.fc1_3 = keras.layers.Dense(65, activation=None)
        self.fc1_4 = keras.layers.Dense(65, activation=None)
        self.fc1_5 = keras.layers.Dense(65, activation=None)
        self.fc1_6 = keras.layers.Dense(65, activation=None)
        self.fc1_7 = keras.layers.Dense(65, activation=None)

    # 前向传播
    def call(self, inputs, training=False):
        # 前向传播
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        fc1_1 = self.fc1_1(x)
        fc1_2 = self.fc1_2(x)
        fc1_3 = self.fc1_3(x)
        fc1_4 = self.fc1_4(x)
        fc1_5 = self.fc1_5(x)
        fc1_6 = self.fc1_6(x)
        fc1_7 = self.fc1_7(x)
        return fc1_1, fc1_2, fc1_3, fc1_4, fc1_5, fc1_6, fc1_7

    @staticmethod
    def calc_loss(logit1, logit2, logit3, logit4, logit5, logit6, logit7, labels):
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)

        # 计算每个位置的损失
        loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit1, labels=labels[:, 0]))
        tf.summary.scalar('loss1', loss1, step=0)

        loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit2, labels=labels[:, 1]))
        tf.summary.scalar('loss2', loss2, step=0)

        loss3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit3, labels=labels[:, 2]))
        tf.summary.scalar('loss3', loss3, step=0)

        loss4 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit4, labels=labels[:, 3]))
        tf.summary.scalar('loss4', loss4, step=0)

        loss5 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit5, labels=labels[:, 4]))
        tf.summary.scalar('loss5', loss5, step=0)

        loss6 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit6, labels=labels[:, 5]))
        tf.summary.scalar('loss6', loss6, step=0)

        loss7 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit7, labels=labels[:, 6]))
        tf.summary.scalar('loss7', loss7, step=0)

        return loss1, loss2, loss3, loss4, loss5, loss6, loss7

    def pred_model(self, logit1, logit2, logit3, logit4, logit5, logit6, logit7, labels):
        logits = [logit1, logit2, logit3, logit4, logit5, logit6, logit7]

        # 获取预测值
        predictions = [tf.argmax(logit, axis=1, output_type=tf.int32) for logit in logits]
        predictions = tf.stack(predictions, axis=1)


        labels = tf.cast(labels, tf.int32)

        # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        return accuracy
