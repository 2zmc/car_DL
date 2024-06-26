import os
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from input_data import OCRIter
from model import OCRModel

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 字符到索引的映射
index = {
    "京": 0, "冀": 1, "津": 2, "晋": 3, "蒙": 4, "辽": 5, "吉": 6, "黑": 7,
    "沪": 8, "苏": 9, "浙": 10, "皖": 11, "闽": 12, "赣": 13, "鲁": 14,
    "豫": 15, "鄂": 16, "湘": 17, "粤": 18, "桂": 19, "琼": 20, "渝": 21,
    "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28,
    "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
    "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44,
    "E": 45, "F": 46, "G": 47, "H": 48, "J": 49, "K": 50, "L": 51, "M": 52,
    "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
    "W": 61, "X": 62, "Y": 63, "Z": 64
}

# 设置图像高度、宽度和批次大小
img_h = 72
img_w = 272
batch_size = 10  # 单张图像预测，批次大小设为1

# 定义模型路径和加载模型
model_path = 'saved_model/zmc_total_model/model_70_90.keras'
model = keras.models.load_model(model_path, custom_objects={'OCRModel': OCRModel})  # 加载最后一次保存的模型

# 创建 OCRIter 实例
data_batch = OCRIter(batch_size, img_h, img_w)

# 获取单张图像和标签
image_batch, label_batch = data_batch.iter()  # 只解包两个值

# 取第一张图像进行预测
single_image = np.array(image_batch[0])

# 预处理图像
processed_image = tf.expand_dims(single_image, axis=0)

# 使用模型进行预测
logits = model(processed_image, training=False)
predicted_label_indices = tf.argmax(logits, axis=-1).numpy()[0]

# 映射预测结果到字符
inv_index = {v: k for k, v in index.items()}  # 索引到字符的映射
predicted_characters = ''.join([inv_index[idx] for idx in predicted_label_indices])

# 输出预测结果
print(f'Predicted characters: {predicted_characters}')

# 使用 matplotlib 绘制图像
plt.figure(figsize=(8, 4))
plt.imshow(single_image, cmap='gray')  # 假设图像是灰度图，使用灰度色彩映射
plt.title(f'识别: {predicted_characters}')
plt.axis('off')
plt.show()
