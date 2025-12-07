# practice1
distinguish cats and dogs
# 猫狗分类器
一个高精度的猫狗图像分类器，使用 PyTorch 和 ResNet18 预训练模型实现。只需将任意猫/狗照片放入指定文件夹，即可获得预测结果。
项目结构
- `data/`: 存放训练和验证数据
- `train/`: 源代码目录
- `requirements.txt`: 依赖列表
- 'predictdata/':放入你想识别的照片
- 'predict.py':识别程序
- 特点
高准确率：在验证集上达到 98.5%+ 的准确率
简单易用：只需将照片放入 predictdata/ 文件夹
自动设备检测：无缝切换 GPU/CPU
路径修复：解决 Windows 常见路径问题
快速预测：单张图片预测时间 < 0.1 秒
无需专业知识：非技术人员也能轻松使用
使用步骤
安装依赖:bash   pip install -r requirements.txt
准备数据集
运行train.py
将照片放入 predictdata/ 文件夹
运行predict.py，获得结果
