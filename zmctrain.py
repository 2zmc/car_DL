import os
import time
import csv
import keras
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from genplate import chars  # 导入字符集合
from input_data import OCRIter  # 导入数据迭代器类
from zmcmodel import CNNModel  # 导入定义的CNN模型类
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 常量和超参数
img_h = 72  # 图像高度
img_w = 272  # 图像宽度
batch_size = 120  # 批次大小
epoch = 1200  # 训练轮数
learning_rate = 0.001  # 学习率
logs_path = 'logs/zmc'  # TensorBoard日志路径
model_path = 'saved_model/zmc'  # 模型保存路径

# 如果目录不存在，则创建目录
os.makedirs(logs_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)


# 图像数据增强函数
def augment_image(image):
    # image = tf.image.random_flip_left_right(image)  # 随机左右翻转
    image = tf.image.random_brightness(image, max_delta=0.1)  # 随机亮度变化
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)  # 随机对比度变化
    image = tf.image.random_hue(image, max_delta=0.1)  # 随机色调变化
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)  # 随机饱和度变化
    return image


# 获取数据批次的函数
def get_batch():
    data_batch = OCRIter(batch_size, img_h, img_w)  # 使用OCRIter类初始化数据迭代器
    image_batch, label_batch = data_batch.iter()  # 获取图像和标签批次
    # image_batch = [augment_image(img) for img in image_batch]  # 对每张图像进行数据增强
    char_to_num = {char: i for i, char in enumerate(chars)}  # 创建字符到数字的映射
    label_batch = [[char_to_num[char] for char in label] for label in label_batch]  # 将标签转换为数字表示
    return np.array(image_batch), np.array(label_batch)  # 返回图像批次和标签批次的numpy数组


# 定义优化器
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# 创建TensorBoard日志写入器
summary_writer = tf.summary.create_file_writer(logs_path)


# 定义训练步骤的函数
@tf.function
def train_step(images, labels, model):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)  # 前向传播计算logits
        losses = model.calc_loss(*logits, labels)  # 计算损失
        total_loss = tf.reduce_sum(losses)  # 计算总损失
    gradients = tape.gradient(total_loss, model.trainable_variables)  # 计算梯度
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # 更新模型参数
    return total_loss, logits


# 定义计算准确率的函数
@tf.function
def compute_accuracy(logits, labels, model):
    accuracy = model.pred_model(*logits, labels)  # 计算准确率
    return accuracy


# 初始化模型
model = CNNModel()
# 次数
metrics_dict = {}
# 初始化损失和准确率的历史记录
loss_history = []
accuracy_history = []


# 训练循环
start_time = time.time()
for step in range(epoch):
    img_batch, lbl_batch = get_batch()  # 获取一个批次的图像和标签数据
    # 执行训练步骤
    total_loss, logits = train_step(img_batch, lbl_batch, model)
    acc = compute_accuracy(logits, lbl_batch, model)
    # 写入TensorBoard日志
    with summary_writer.as_default():
        tf.summary.scalar('loss', total_loss, step=step)  # 记录损失
        tf.summary.scalar('accuracy', acc, step=step)  # 记录准确率
        tf.summary.image('input_image', img_batch, step=step)  # 记录输入图像
        tf.summary.scalar("epoch", step + 1, step=step)  # 记录当前训练轮次
    # 打印训练进度
    if step % 1 == 0:
        print(f'第 {step} 步, 损失: {total_loss.numpy()}, 准确率: {acc.numpy() * 100:.2f}%')
    # 记录每个step的指标
    metrics_dict[step] = {
        'loss': total_loss.numpy(),
        'accuracy': acc.numpy() * 100,
        'step_accuracy': acc.numpy()
    }
    # 记录历史损失和准确率
    loss_history.append(total_loss.numpy())
    accuracy_history.append(acc.numpy() * 100)
    # 每50步保存一次模型或者在最后一步保存
    if step % 50 == 0 or step == epoch - 1:
        model.save(os.path.join(model_path, f'model_{step}.keras'))


# 保存数据到CSV文件
def log_metrics_to_csv(metrics_dict, csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['步骤', '损失', '准确率', '步骤准确率'])
        for step, metrics in metrics_dict.items():
            writer.writerow([step, metrics['loss'], metrics['accuracy'], metrics['step_accuracy']])


# 调用函数保存到CSV文件
log_metrics_to_csv(metrics_dict, 'training_metrics.csv')


# 可视化损失和准确率随epoch变化的图表
def visualize_metrics(loss_history, accuracy_history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='损失', marker='o')
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, label='准确率', marker='o')
    plt.title('训练准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 调用函数进行可视化
visualize_metrics(loss_history, accuracy_history)

# 训练完成
end_time = time.time()
print(f"训练完成. 耗时: {end_time - start_time:.2f} 秒")
