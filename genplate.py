import os  # 可以方便地进行文件和目录操作、环境变量访问、进程管理、系统信息获取以及文件权限和属性修改等操作
import cv2 as cv  # 可以帮助开发者进行图像和视频的读取、显示、处理、分析和识别。
import numpy as np  # 是Python 语言的一个第三方库，其支持大量高维度数组与矩阵运算。此外，NumPy 也针对数组运算提供大量的数学函数
from math import *  # 数学运算中除了一些基本运算以外，还支持一些特殊运算，如求绝对值、阶乘、最大公约数等
from PIL import ImageFont, Image, ImageDraw  # 图像处理
import matplotlib
matplotlib.use('TkAgg')  # 或者其他支持的后端，如 'Qt5Agg', 'WXAgg' 等
# Image用于创建、打开、保存和操作图像
# ImageFont 模块用于加载和操作字体文件，以便在图像上进行文字绘制
# ImageDraw 模块提供了在图像上绘制图形和文本的功能


# 可以省略
from matplotlib import pyplot as plt

index = {
    "京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "豫": 5, "云": 6, "辽": 7, "黑": 8, "湘": 9,
    "皖": 10, "鲁": 11, "新": 12, "苏": 13, "浙": 14, "赣": 15, "鄂": 16, "桂": 17, "甘": 18, "晋": 19,
    "蒙": 20, "陕": 21, "吉": 22, "闽": 23, "贵": 24, "粤": 25, "青": 26, "藏": 27, "川": 28, "宁": 29,
    "琼": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40,
    "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48, "J": 49, "K": 50,
    "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
    "W": 61, "X": 62, "Y": 63, "Z": 64}  # 创建字典以及索引

chars = ['京', '沪', '津', '渝', '冀', '豫', '云',
         '辽', '黑', '湘', '皖', '鲁', '新', '苏',
         '浙', '赣', '鄂', '桂', '甘', '晋', '蒙',
         '陕', '吉', '闽', '贵', '粤', '青', '藏', '川', '宁', '琼',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
         'Y', 'Z']


# 创建列表

# 生成字符图像
# 大小45*70白色背景并在其上用字体f绘制黑色字符val。
# 将图像调整为宽23像素的大小并返回图像的numpy数组表示。
def GenCh(f, val):
    img = Image.new("RGB", (45, 70), (255, 255, 255))  # 创建一个新的图像
    draw = ImageDraw.Draw(img)  # 绘图功能
    draw.text((0, 3), val, (0, 0, 0), font=f)  # 文本导入图片当中
    img = img.resize((23, 70))  # 变换图像大小为23-70 resize(width, height)
    A = np.array(img)  # 转成数组形式
    return A  # 返回


# 生成一个宽23像素，高70像素的白色背景图像
# 并在其上用字体f绘制黑色字符val。返回图像的numpy数组表示。
def GenCh1(f, val):
    img = Image.new("RGB", (23, 70), (255, 255, 255))  # 创建一个新的图像
    draw = ImageDraw.Draw(img)  # 绘图功能
    draw.text((0, 2), val, (0, 0, 0), font=f)  # 文本导入图片当中
    A = np.array(img)  # 转成数组形式
    return A  # 返回


# 生成随机数字
def r(val):
    return int(np.random.random() * val)  # 随机生成数字


# img旋转变换, angel角度变换, shape形状尺寸
# max_angel最大旋转角度
# 通过定义四个点来进行实现透视变换
def rot(img, angel, shape, max_angel):  # img输入图片 angel旋转角度 shape图像形状 max_angel最大旋转角度
    size_o = [shape[1], shape[0]]  # 定义一个列表包含宽度和高度 shape宽度 shape0高度
    size = (shape[1] + int(shape[0] * cos((float(max_angel) / 180) * 3.14)), shape[0])  # 变换过后的图像的宽度以及高度，高度保持不变
    interval = abs(int(sin((float(angel) / 180) * 3.14) * shape[0]))  # 计算旋转引起的水平偏移量  sin()函数计算旋转角度过后的偏移量
    pts1 = np.float32([[0, 0], [0, size_o[1]], [size_o[0], 0], [size_o[0], size_o[1]]])  # 4-2的数组表示图像的四个点坐标
    # 下面如果角度大于0顺时针旋转  如果角度小于0逆时针旋转
    if angel > 0:
        pts2 = np.float32([[interval, 0], [0, size[1]], [size[0], 0], [size[0] - interval, size_o[1]]])
    else:
        pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0], [size[0], size_o[1]]])
    M = cv.getPerspectiveTransform(pts1, pts2)  # 使用OpenCV里面getPerspectiveTransform()函数计算透视变换
    dst = cv.warpPerspective(img, M, size)  # 使用OpenCV里面warpPerspective()函数将透视图像输出到img里面得到dst
    return dst  # 指定输出图像的尺寸


# 对图像img进行透视变换，变换范围由factor决定，size为图像尺寸
def rotRandrom(img, factor, size):  # img输入图片 factor控制随机变换的因子 size为图像尺寸
    shape = size  # 用于储存图像的形状（宽--高）
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])  # [0, shape[0]],
    # 旋转图片，左上角，左下角，右上角，右下角
    pts2 = np.float32([[r(factor), r(factor)], [r(factor), shape[0] - r(factor)],
                       [shape[1] - r(factor), r(factor)], [shape[1] - r(factor), shape[0] - r(factor)]])  #
    # 加上一个随机数字变换r()函数
    M = cv.getPerspectiveTransform(pts1, pts2)  # 计算透视从pts1-pts2的矩阵变换
    dst = cv.warpPerspective(img, M, size)  # 使用OpenCV里面warpPerspective()函数将透视图像输出到img里面得到dst
    return dst


# 添加污点 向图像img中添加污点，污点图像来源于sum
def Addsmudginess(img, Smu):  # 输入图像与污点图片
    rows = r(Smu.shape[0] - 50)  # 污点高度生成随机的整数
    cols = r(Smu.shape[1] - 50)  # 污点宽度
    adder = Smu[rows:rows + 50, cols:cols + 50]  # 提取一个50*50的区域
    adder = cv.resize(adder, (350, 350))  # 将图像调整resize(输入图像，图像尺寸)
    img = cv.resize(img, (350, 350))
    # dst = cv.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
    # src: 输入图像UTF-8 dsize: 输出图像的尺寸
    img = cv.bitwise_not(img)  # 颜色反转 黑变白  白变黑
    img = cv.bitwise_and(adder, img)
    img = cv.bitwise_not(img)  # 颜色反转 黑变白  白变黑
    return img


# 随机调整图像img的色调，饱和度和亮度
def tfactor(img):  # 输入图片进行调整
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 将图像从RGB转换为HSV颜色空间
    hsv[:, :, 0] = hsv[:, :, 0] * (0.8 + np.random.random() * 0.2)  # 随机调整颜色
    hsv[:, :, 1] = hsv[:, :, 1] * (0.3 + np.random.random() * 0.7)  # 随机调整饱和度
    hsv[:, :, 2] = hsv[:, :, 2] * (0.2 + np.random.random() * 0.8)  # 随机调整亮度
    img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)  # 将图像转换会RGB颜色空间
    return img


# 用于将图像img放置在一个随机选择的无车牌的背景图上面
# 背景调整到与图像相同大小
def random_envirment(img, noplate_bg):  # img车牌图片 noplate_bg包含无车牌背景图片路径
    bg_index = r(len(noplate_bg))  # 列宽长款范围类随机一个数字作为背景图索引
    env = cv.imread(noplate_bg[bg_index])  # 使用OpenCV里面imread()函数读取无车牌背景图
    env = cv.resize(env, (img.shape[1], img.shape[0]))  # 使用resize()函数调整输入的img与之尺寸相同
    bak = (img == 0)  # 创建一个掩码 bak，该掩码的值在 img 中为黑色（即值为0）的位置为 True，其他位置为 False。
    bak = bak.astype(np.uint8) * 255  # 转换成uint8  出于255变换成2进制格式
    inv = cv.bitwise_and(bak, env)  # 将掩码 bak 应用到背景图像 env。结果是 env 中掩码为白色（255）的位置保留原背景图像的像素值，其他位置变为黑色。
    img = cv.bitwise_or(inv, img)  # 将处理后的背景图像 inv 与输入图像 img 合并。结果是 img 中原本为黑色的区域现在填充为背景图像的相应部分，其他区域保持原样。
    return img


# 添加高斯模糊，模糊程度有level决定
def AddGauss(img, level):
    return cv.blur(img, (level * 2 + 1, level * 2 + 1))


class GenPlate:
    # 初始化函数，加载字体，模板背景和图像污点图像
    # 并且收集无背景的背景图片路径
    def __init__(self, fontch, fontEng, NoPlates):  # fontch中文字体路径  fontEng英文字体luj   无车牌背景的目录
        self.fontc = ImageFont.truetype(fontch, 43, 0)  # 使用 PIL 库的 ImageFont.truetype 方法加载中文字体文件，字体大小设置为43。
        self.fontE = ImageFont.truetype(fontEng, 60, 0)  # 使用 PIL 库的 ImageFont.truetype 方法加载英文字体文件，字体大小设置为60。
        self.img = np.array(Image.new("RGB", (226, 70), (255, 255, 255)))  # 创建空白图片背景色白色  转换成数组
        # sum_img = ["images/b1.bmp", "images/g1.jpg", "images/y1.bmp",
        #            "images/b2.bmp", "images/g2.jpg", "images/y1.bmp"]
        self.bg = cv.resize(cv.imread("images/b1.bmp"), (226, 70))  # 加载背景模板
        self.smu = cv.imread("images/smu2.jpg")  # 加载污点模板图片
        self.noplates_path = []  # 创建一个空列表 self.noplates_path，用于存储无车牌背景图像的路径。
        for parent, parent_folder, filenames in os.walk(NoPlates):  # 遍历无车牌背景图目录
            for filename in filenames:  # parent：当前目录路径
                path = parent + "/" + filename  # parent_folder：子目录列表。
                self.noplates_path.append(path)  # filenames：当前目录中的文件列表。

    # 在模板背景上绘制车牌号
    def draw(self, val):  # val这是一个包含车牌号各个字符的列表或字符串。
        offset = 2  # 偏移量  设置为2，用于调整字符在图像中的位置。
        self.img[0:70, offset + 8:offset + 8 + 23] = GenCh(self.fontc, val[0])
        self.img[0:70, offset + 8 + 23 + 6:offset + 8 + 23 + 6 + 23] = GenCh1(self.fontE, val[1])
        for i in range(5):
            base = offset + 8 + 23 + 6 + 23 + 17 + i * 23 + i * 6
            self.img[0:70, base:base + 23] = GenCh1(self.fontE, val[i + 2])
        return self.img

    # 生成车牌图像。根据输入车牌号码text，通过一系列图像处理步骤生成最终的车牌图像
    def generate(self, text):  # generate 方法接受一个参数 text，这是一个长度为7的字符串，表示车牌号。
        if len(text) == 7:  # 只有当输入的 text 长度为7时，才继续执行下面的代码。否则，不进行任何操作。
            #  绘制车牌：
            fg = self.draw(text)  # 调用 draw 方法生成车牌号图像 fg。
            fg = cv.bitwise_not(fg)  # 使用 cv.bitwise_not 对图像进行颜色反转，将黑色变为白色，白色变为黑色。
#  将车牌号叠加到背景上
            com = cv.bitwise_or(fg, self.bg)  # 使用 cv.bitwise_or 将反转后的车牌号图像 fg 叠加到背景图像 self.bg 上。
            # plt.imshow(com, cmap='gray')  # 显示图像，使用灰度颜色映射
            # plt.show()  # 显示图像窗口
            com = rot(com, r(60) - 30, com.shape, 30)
            com = rotRandrom(com, 10, (com.shape[1], com.shape[0]))
            # plt.imshow(com, cmap='gray')  # 显示图像，使用灰度颜色映射
            # plt.show()  # 显示图像窗口
            com = Addsmudginess(com, self.smu)  # 调用 Addsmudginess 方法向图像 com 添加污点，污点图像来源于 self.smu。
            com = tfactor(com)
            com = random_envirment(com, self.noplates_path)  # 添加随机背景
            com = AddGauss(com, 1 + r(4))  # 添加高斯模糊
            # plt.imshow(com)  # 显示图像，使用灰度颜色映射
            # plt.show()  # 显示图像窗口

            return com  # 返回已经处理过的车牌图像 com。

    @staticmethod
    # 生成一个随机车牌号码字符串，并返回字符串和其列表表示。
    def genPlatestring(pos, val):  # pos整数 需要替换掉位置 不替换-1  val要替换的字符
        plateStr = ""  # 储存车牌字符串
        plateList = []  # 储存每个字符的中间结果
        box = [0, 0, 0, 0, 0, 0, 0]  # 车牌长度
        if pos != -1:
            box[pos] = 1
        for unit, cpos in zip(box, range(len(box))):
            if unit == 1:
                plateStr += val
                plateList.append(val)
            else:
                if cpos == 0:
                    plateStr += chars[r(31)]  # 省份
                    plateList.append(plateStr)
                elif cpos == 1:
                    plateStr += chars[41 + r(24)]  # 英文字母
                    plateList.append(plateStr)
                else:
                    plateStr += chars[31 + r(34)]  # 英文字母或者数字
                    plateList.append(plateStr)
        plate = [plateList[0]]  # 初始化列表
        b = [plateList[i][-1] for i in range(len(plateList))]  # 包含字符串的最后一个字符
        plate.extend(b[1:7])  # 将 b 列表中从第2个到第7个元素（索引1到6）添加到 plate 中。
        return plateStr, plate

    @staticmethod
    # 批量生成车牌图片，根据指定的批量大小batchsize，输出路径outputPath
    # 和图像尺寸size，生成指定的数量的车牌并且保存到指定的路径
    def genBatch(batchsize, outputPath, size):
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)  # # 如果输出路径不存在，则创建该路径
        outfile = open('label.txt', 'w', encoding='utf-8')  # 打开一个文件用于写入标签信息
        for i in range(batchsize):
            plateStr, plate = GenPlate.genPlatestring(-1, -1)   # 生成车牌号字符串和字符列表
            print(plateStr, plate)  # 打印生成的车牌号字符串和字符列表
            img = G.generate(plateStr)

            img = cv.resize(img, size)

            # 将生成的图片保存到指定路径，文件名格式为两位数的序号加上.jpg后缀
            cv.imwrite(outputPath + "/" + str(i).zfill(2) + ".jpg", img)
            outfile.write(str(plate) + "\n")


# 主程序入口，创建GenPlate对象，并调用genBatch函数生成30张车牌图像，
# 保存到“plate”文件夹下，并将标签保存到“label.txt”文件中。
if __name__ == '__main__':
    G = GenPlate('font/platech.ttf', 'font/platechar.ttf', 'NoPlates')
    G.genBatch(3, "plate", (272, 72))  # 生成300


# 2.16.1

