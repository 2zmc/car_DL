import os
import keras
import numpy as np
import tensorflow as tf
from input_data import OCRIter
from model import OCRModel, calc_loss, accuracy_fn
from genplate import chars  # 假设chars是所有可能字符的集合

# 设置图像高度、宽度、标签数量和批次大小
img_h = 72
img_w = 272
num_label = 7
batch_size = 30
num_test_batches = 5  # 假设测试数据总数目约为 batch_size * num_test_batches

# 定义模型路径和加载模型
model = keras.models.load_model('saved_model/zmc/model_1199.keras',
                                custom_objects={'OCRModel': OCRModel})  # 加载最后一次保存的模型

# 创建字符到数字的映射
char_to_num = {char: i for i, char in enumerate(chars)}


# 获取测试批次数据的函数
def get_test_data():
    data_batch = OCRIter(batch_size, img_h, img_w)
    all_test_images = []
    all_test_labels = []
    for _ in range(num_test_batches):
        image_batch, label_batch = data_batch.iter()  # 修改为只解包两个值
        all_test_images.append(np.array(image_batch))
        # 将标签转换为数字表示
        numeric_labels = [[char_to_num[char] for char in label] for label in label_batch]
        all_test_labels.append(np.array(numeric_labels, dtype=np.int32))
    return np.concatenate(all_test_images), np.concatenate(all_test_labels)


# 初始化损失和准确率度量
test_loss_metric = keras.metrics.Mean(name='test_loss')
test_accuracy_metric = keras.metrics.Mean(name='test_accuracy')


def test_step(images, labels):
    logits = model(images, training=False)
    losses = calc_loss(*logits, labels)
    total_loss = tf.reduce_sum(losses)
    acc = accuracy_fn(logits, labels)
    test_loss_metric(total_loss)
    test_accuracy_metric(acc)
    return logits, losses, total_loss, acc


# 获取测试数据
test_images, test_labels = get_test_data()

# 执行测试步骤
logits, losses, total_loss, acc = test_step(test_images, test_labels)

# 打印测试结果
print(f'Test Loss: {test_loss_metric.result():.4f}')
print(f'Test Accuracy: {test_accuracy_metric.result() * 100:.2f}%')

# 重置指标
test_loss_metric.reset_state()
test_accuracy_metric.reset_state()
