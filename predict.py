import os
import json

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from nets.Alexnet import AlexNet_v1, AlexNet_v2
from nets.Vgg import vgg
from nets.Googlenet import GoogLeNet
from nets.Resnet import resnet50
from nets.Mobilenetv2 import MobileNetV2
from nets.Mobilenetv3 import mobilenet_v3_small,mobilenet_v3_large
from nets.Shufflenet import shufflenet_v2_x1_0
from nets.EfficientnetV1 import efficientnet_b0
from nets.EfficientnetV2 import efficientnetv2_s


# -------------------------------- 训练参数配置 --------------------------------------# 
im_height = 224 # 输入图片的高
im_width = 224  # 输入图片的宽
model_name = "AlexNet" # 模型名称"AlexNet"、"VGG"、"Googlenet","Resnet","MobilenetV3"、"Shufflenet"
img_path = r"fish_data\test\longyu.jpg" # 预测的图片
# ------------------------------------------------------------------------------------# 
# 加载图片
assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
img = Image.open(img_path)

# 图片修改为 224x224
img = img.resize((im_width, im_height))
plt.imshow(img)

# 归一化
img = np.array(img) / 255.

# 添加一个batch_size维度
img = (np.expand_dims(img, 0))

# 读取种类字典
json_path = './class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

with open(json_path, "r") as f:
    class_indict = json.load(f)

# 实例化模型
if model_name == "AlexNet":
    model = AlexNet_v1(im_height=im_height, im_width=im_width, num_classes=9)
elif model_name == "VGG":
    model = vgg(model_name="vgg16", im_height=224, im_width=224, num_classes=9)
elif model_name == "Googlenet":
    model = GoogLeNet(class_num=9, aux_logits=False)    
elif model_name == "Resnet":
    model = resnet50(num_classes=9, include_top=True)
# elif model_name == "MobilenetV2":
#     model = MobileNetV2(im_height=im_height, im_width=im_width, num_classes=9)
elif model_name == "MobilenetV3":
    model = mobilenet_v3_large(input_shape=(im_height, im_width, 3),
                               num_classes=9,
                               include_top=True)    
elif model_name == "Shufflenet":
    model = shufflenet_v2_x1_0(input_shape=(im_height, im_width, 3),
                               num_classes=9)

weighs_path = f"./save_weights/my{model_name}.h5"
assert os.path.exists(img_path), "file: '{}' dose not exist.".format(weighs_path)
model.load_weights(weighs_path, by_name=True)

# prediction
result = np.squeeze(model.predict(img))
predict_class = np.argmax(result)

print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                result[predict_class])
plt.title(print_res)
for i in range(len(result)):
    print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                result[i]))
plt.show()
