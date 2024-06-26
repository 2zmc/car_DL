import tensorflow as tf
import keras


class OCRModel(keras.Model):
    def __init__(self):
        super(OCRModel, self).__init__()
        # 第一个卷积层，卷积核大小为3x3，32个神经元，填充方式为：valid,激活函数为ReLU
        self.conv1 = keras.layers.Conv2D(32, (3, 3), padding='valid', activation='relu')
        # 第二个卷积层，卷积核大小为3x3，32个神经元，填充方式为：valid，激活函数为ReLU
        self.conv2 = keras.layers.Conv2D(32, (3, 3), padding='valid', activation='relu')
        # 第一个池化层，池化窗口大小为2x2，步长为2,填充方式为：valid
        self.pool1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid')
        # 第三个卷积层，卷积核大小为3x3，64个卷积核，激活函数为ReLU
        self.conv3 = keras.layers.Conv2D(64, (3, 3), padding='valid', activation='relu')
        # 第四个卷积层，卷积核大小为3x3，64个卷积核，激活函数为ReLU
        self.conv4 = keras.layers.Conv2D(64, (3, 3), padding='valid', activation='relu')
        # 第二个池化层，池化窗口大小为2x2，步长为2
        self.pool2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid')
        # 第五个卷积层，卷积核大小为3x3，128个卷积核，激活函数为ReLU
        self.conv5 = keras.layers.Conv2D(128, (3, 3), padding='valid', activation='relu')
        # 第六个卷积层，卷积核大小为3x3，128个卷积核，激活函数为ReLU
        self.conv6 = keras.layers.Conv2D(128, (3, 3), padding='valid', activation='relu')
        # 第三个池化层，池化窗口大小为2x2，步长为2
        self.pool3 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid')
        # 展平层，将三维的特征图展平成一维向量
        self.flatten = keras.layers.Flatten()
        # Dropout层，防止过拟合，丢弃40%的节点
        self.drop = keras.layers.Dropout(0.4)
        # 全连接层列表，包含7个全连接层，每个层输出65个神经元
        self.fc_layers = [keras.layers.Dense(65) for _ in range(7)]

    # 定义前向传播过程
    def call(self, inputs, training=False):
        # 通过第一个卷积层
        x = self.conv1(inputs)
        # 通过第二个卷积层
        x = self.conv2(x)
        # 通过第一个池化层
        x = self.pool1(x)
        # 通过第三个卷积层
        x = self.conv3(x)
        # 通过第四个卷积层
        x = self.conv4(x)
        # 通过第二个池化层
        x = self.pool2(x)
        # 通过第五个卷积层
        x = self.conv5(x)
        # 通过第六个卷积层
        x = self.conv6(x)
        # 通过第三个池化层
        x = self.pool3(x)
        # 展平层，将三维的特征图展平成一维向量
        x = self.flatten(x)
        # 如果是在训练过程中，则应用Dropout
        if training:
            x = self.drop(x, training=training)
        # 通过7个全连接层，获得7个不同位置字符的预测结果
        logits = [fc(x) for fc in self.fc_layers]
        return logits


# 定义计算损失函数
def calc_loss(logit1, logit2, logit3, logit4, logit5, logit6, logit7, labels):
    # 将标签转换为张量
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    # 计算每个位置的交叉熵损失，并添加到TensorBoard的摘要中
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

    # 返回所有位置的损失值
    return loss1, loss2, loss3, loss4, loss5, loss6, loss7


# 定义计算准确率的函数
def accuracy_fn(logits, labels):
    # 计算每个位置的预测结果
    predictions = [tf.argmax(logit, axis=1, output_type=tf.int32) for logit in logits]
    # 计算每个位置预测正确的数量
    correct_predictions = [tf.equal(predictions[i], labels[:, i]) for i in range(7)]
    # 计算总体准确率，即所有位置都正确的数量
    accuracy = tf.reduce_mean(tf.cast(tf.reduce_all(correct_predictions, axis=0), tf.float32))
    return accuracy
