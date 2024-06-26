from genplate import *  # 导入自定义的车牌生成模块
import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图
import matplotlib
matplotlib.use('TkAgg')  # 或者其他支持的后端，如 'Qt5Agg', 'WXAgg' 等


# 用于生成车牌图像的函数
class OCRIter:
    def __init__(self, batch_size, width, height):
        super(OCRIter, self).__init__()
        self.genplate = GenPlate("font/platech.ttf", "font/platechar.ttf", "NoPlates")  # 字体
        self.batch_size = batch_size  # 批量大小
        self.height = height  # 图像高度
        self.width = width  # 图像宽度

    def iter(self):
        data = []  # 用于存储生成的图像数据
        label = []  # 用于存储对应的标签
        for i in range(self.batch_size):  # 循环生成 batch_size 个图像和标签
            img, num = self.gen_sample(self.genplate, self.width, self.height)  # 生成单个车牌图像和标签
            data.append(img)  # 将图像添加到数据列表
            label.append(num)  # 将图像添加到 标签列表
        return np.array(data), np.array(label)  # 返回包含所有图像和标签的 numpy 数组

    # 静态方法，用于生成指定范围内的随机数
    @staticmethod
    def rand_range(lo, hi):
        return lo + r(hi - lo)   # 调用r()函数返回指定范围内的随机数

    def gen_rand(self):
        name = ""  # 初始化车牌号码字符串
        label = list([])   # 初始化标签列表
        label.append(self.rand_range(0, 31))  # 随机生成车牌的第一个字符（省份）
        label.append(self.rand_range(41, 65))   # 随机生成车牌的第二个字符（字母）
        for i in range(5):  # 随机生成车牌后五个字符（字母或数字）
            label.append(self.rand_range(31, 65))
        name += chars[label[0]]  # 将第一个字符添加到车牌号码
        name += chars[label[1]]  # 将第二个字符添加到车牌号码
        for i in range(5):
            name += chars[label[i+2]]  # 将最后五个添加到车牌号码
        return name, label  # 返回生成的车牌号码和标签

    # 生成单个车牌图像和标签
    def gen_sample(self, genplate, width, height):
        num, label = self.gen_rand()  # 生成随机车牌号码和标签
        img = genplate.generate(num)  # 根据车牌号码生成车牌图像
        # print(img.shape)  # 打印图像形状
        # print(img.dtype)  # 打印图像数据类型
        img = cv.resize(img, (height, width))  # 调整图像大小
        #plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # OpenCV 读取的图像需要转换颜色空间才能在 Matplotlib 中正确显示
        #plt.show()
        img = np.multiply(img, 1/255.0)  # 将图像像素值归一化到 [0, 1] 之间
        return img, num  # 返回生成的图像和车牌号码


# 示例用法，展示如何使用 OCRIter 类生成车牌图像和标签
if __name__ == "__main__":
    o = OCRIter(1, 72, 272)   # 创建 OCRIter 类的实例，批量大小为 10，图像尺寸为 72x272
    img, lbl = o.iter()  # 生成一批车牌图像和标签
    for im in img:  # 遍历每个生成的图像
        plt.imshow(im)   # 显示图像，使用灰度颜色映射
        plt.show()  # 显示图像窗口
    print(img.shape)  # 打印生成的图像数据的形状
    print(lbl)  # 打印生成的标签
