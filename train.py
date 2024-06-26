import os
import keras
import time
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
from genplate import chars
from input_data import OCRIter
from model import OCRModel, calc_loss, accuracy_fn

# 设置日志级别，避免过多的日志输出
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


# 设置图像高度、宽度、标签数量、批次大小、训练轮数和学习率
img_h = 72
img_w = 272
num_label = 7
batch_size = 140
epoch = 5
learning_rate = 0.001  # 降低学习率以提高稳定性

# 定义日志路径和模型保存路径
logs_path = 'logs'
model_path = 'saved_model'


# 获取批次数据的函数
def get_batch():
    data_batch = OCRIter(batch_size, img_h, img_w)  # 实例化OCRIter类以获取数据
    image_batch, label_batch = data_batch.iter()  # 获取图像和标签批次
    char_to_num = {char: i for i, char in enumerate(chars)}  # 创建字符到数字的映射
    label_batch = [[char_to_num[char] for char in label] for label in label_batch]  # 将标签转换为数字表示
    return np.array(image_batch), np.array(label_batch)  # 返回图像批次和标签批次的numpy数组


# 初始化模型和优化器
model = OCRModel()  # 实例化OCR模型
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)  # 使用Adam优化器
loss_metric = keras.metrics.Mean(name='train_loss')  # 记录训练损失的指标
accuracy_metric = keras.metrics.Mean(name='train_accuracy')  # 记录训练准确率的指标


# 定义单步训练函数
def train_step(images, labels):
    with tf.GradientTape() as tape:  # 使用tf.GradientTape记录梯度信息
        logits = model(images, training=True)  # 前向传播，获取预测结果
        losses = calc_loss(*logits, labels)  # 计算损失
        total_loss = tf.reduce_sum(losses)  # 总损失为所有位置的损失之和
    gradients = tape.gradient(total_loss, model.trainable_variables)  # 计算梯度
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # 应用梯度更新模型参数
    acc = accuracy_fn(logits, labels)  # 计算准确率
    loss_metric(total_loss)  # 更新损失指标
    accuracy_metric(acc)  # 更新准确率指标
    return logits, losses, total_loss, acc  # 返回预测结果、损失、总损失和准确率


# 初始化TensorBoard记录器
train_writer = tf.summary.create_file_writer(logs_path)


# 初始化变量以跟踪最佳精度和最低损失
best_acc = 0.0
lowest_loss = float('inf')


# 记录训练开始时间
start_time1 = time.time()
for epoch_num in range(epoch):
    img_batch, lbl_batch = get_batch()  # 获取批次数据
    start_time2 = time.time()  # 记录每个step开始时间
    time_str = datetime.datetime.now().isoformat()  # 获取当前时间

    # 绘制并显示第一张图像及其对应的标签
    first_image = img_batch[0]
    first_label = lbl_batch[0]
    plt.figure()
    plt.imshow(first_image)
    plt.title(f'Epoch {epoch_num}, Label: {first_label}')
    plt.axis('off')
    plt.show()

    logits, losses, total_loss, acc = train_step(img_batch, lbl_batch)  # 执行单步训练

    # 记录摘要信息到TensorBoard
    with train_writer.as_default():  # 计算每个step的持续时间
        tf.summary.scalar('loss', loss_metric.result(), step=epoch_num)
        tf.summary.scalar('accuracy', accuracy_metric.result(), step=epoch_num)

    duration = time.time() - start_time2  # 计算每个step的持续时间
    if epoch_num % 1 == 0:  # 每1个step打印一次日志
        print(
            f'{time_str}: epoch {epoch_num}, loss_total = {loss_metric.result():.2f}, acc = {accuracy_metric.result() * 100:.2f}%, sec/batch = {duration:.2f}')

    if epoch_num % 10 == 0:
        acc_value = acc.numpy()  # 将 TensorFlow 张量转换为 NumPy 数组
        print(f"Epoch {epoch_num}: 样本准确率 - {acc_value:.4f}")

    # 如果当前精度高于已记录的最佳精度，则保存模型
    if acc > best_acc:
        best_acc = acc
        model.save(os.path.join(model_path, 'model_best_accuracy.keras'))
        print(f"New best accuracy model saved at epoch {epoch_num} with accuracy {best_acc:.4f}")

    # 如果当前损失低于已记录的最低损失，则保存模型
    if total_loss < lowest_loss:
        lowest_loss = total_loss
        model.save(os.path.join(model_path, 'model_lowest_loss.keras'))
        print(f"New lowest loss model saved at epoch {epoch_num} with loss {lowest_loss:.4f}")

    if epoch_num % 50 == 0 or (epoch_num + 1) == epoch:  # 每50个step或在最后一个step保存一次模型
        model.save(os.path.join(model_path, f'model_epoch_{epoch_num}.keras'))

    # 重置指标
    loss_metric.reset_state()
    accuracy_metric.reset_state()

end_time = time.time()  # 记录训练结束时间
print("训练结束，共耗时 {:.2f} 分钟".format((end_time - start_time1) / 60))  # 输出总耗时
model.summary()  # 打印模型摘要



