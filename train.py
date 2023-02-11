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

import matplotlib.pyplot as plt
import tensorflow as tf
import json
import os,math
import time
import glob
import random
import logging

# -------------------------------- 训练参数配置 --------------------------------------# 
im_height = 224 # 输入图片的高
im_width = 224  # 输入图片的宽
num_classes = 9   # 识别的种类
batch_size = 4 # 训练的批次图片数
epochs = 50     # 训练迭代次数
model_name = "Shufflenet" # 模型名称"AlexNet"、"VGG"、"Googlenet","Resnet","MobilenetV2","MobilenetV3"、"Shufflenet"、"EfficientnetV1","EfficientnetV2"
log_draw_loss_suc = True # 是否需要通过日志绘图
# ------------------------------------------------------------------------------------# 


# -------------------------------- 调用GPU -------------------------------------------# 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        exit(-1)
# ------------------------------------------------------------------------------------# 

# -------------------------------------日志模块----------------------------------------#
# 创建logger对象
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # log等级总开关
# log输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
# 控制台handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO) # log等级的开关
stream_handler.setFormatter(formatter)
# 文件handler
file_handler = logging.FileHandler(f"logs/{model_name}.log",encoding='utf-8')
file_handler.setLevel(logging.INFO) # log等级的开关
file_handler.setFormatter(formatter)
# 添加到logger
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
logger.info("训练开始!")
# -------------------------------------------------------------------------------------#
    

# 数据集地址
data_root = os.path.abspath(os.getcwd())  # get data root path
image_path = os.path.join(data_root, "fish_data")  # flower data set path
train_dir = os.path.join(image_path, "train")
validation_dir = os.path.join(image_path, "val")
# 判断是否存在数据集文件
assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

# 创建存储模型的文件夹
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")

# 获取种类字典
data_class = [cla for cla in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cla))]
class_num = len(data_class)
class_dict = dict((value, index) for index, value in enumerate(data_class))
    
# 种类字典的键和值转换
inverse_dict = dict((val, key) for key, val in class_dict.items())
# 保存字典
json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)
    
    
# 加载训练集
train_image_list = glob.glob(train_dir+"/*/*.jpg")
random.shuffle(train_image_list)
train_num = len(train_image_list)
assert train_num > 0, "cannot find any .jpg file in {}".format(train_dir)
train_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in train_image_list]

# 加载验证集
val_image_list = glob.glob(validation_dir+"/*/*.jpg")
random.shuffle(val_image_list)
val_num = len(val_image_list)
assert val_num > 0, "cannot find any .jpg file in {}".format(validation_dir)
val_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in val_image_list]

logger.info("using {} images for training, {} images for validation.".format(train_num,val_num))
# 预处理                
def process_path(img_path, label):
    label = tf.one_hot(label, depth=class_num)
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [im_height, im_width])[...,:3]
    
    return image, label    

AUTOTUNE = tf.data.experimental.AUTOTUNE # 数据读取流水线加速

# load train dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
train_dataset = train_dataset.shuffle(buffer_size=train_num)\
                                .map(process_path, num_parallel_calls=AUTOTUNE)\
                                .repeat().batch(batch_size).prefetch(AUTOTUNE)

# load train dataset
val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, val_label_list))
val_dataset = val_dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                            .repeat().batch(batch_size)
       
                         
# -------------------------------------- 实例化模型 ------------------------------------------------------#
if model_name == "AlexNet":
    model = AlexNet_v1(im_height=im_height, im_width=im_width, num_classes=num_classes)
    # model = AlexNet_v2(class_num=num_classes)
    # model.build((batch_size, 224, 224, 3))  # when using subclass model
elif model_name == "VGG":
    model = vgg(model_name="vgg16", im_height=im_height, im_width=im_width, num_classes=num_classes)
elif model_name == "Googlenet":
    model = GoogLeNet(im_height=im_height, im_width=im_width, class_num=num_classes, aux_logits=True)
elif model_name == "Resnet":
    model = resnet50(im_height=im_height, im_width=im_width, num_classes=num_classes, include_top=True)
    pre_weights_path = './checkpoints/tf_resnet50_weights/pretrain_weights.ckpt'
    assert len(glob.glob(pre_weights_path + "*")), "cannot find {}".format(pre_weights_path)
    model.load_weights(pre_weights_path)
elif model_name == "MobilenetV2":
    model = MobileNetV2(im_height=im_height, im_width=im_width, num_classes=num_classes)
    pre_weights_path = './checkpoints/tf_mobilenetv2_weights/pretrain_weights.ckpt'
    assert len(glob.glob(pre_weights_path+"*")), "cannot find {}".format(pre_weights_path)
    model.load_weights(pre_weights_path)
elif model_name == "MobilenetV3":
    model = mobilenet_v3_large(input_shape=(im_height, im_width, 3),
                               num_classes=num_classes,
                               include_top=True)
    # pre_weights_path = './checkpoints/mobilenet_v3_large_224_1.0.h5'
    # assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)
    # model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)
elif model_name == "Shufflenet":
    model = shufflenet_v2_x1_0(input_shape=(im_height, im_width, 3),
                               num_classes=num_classes)
    # pre_weights_path = './checkpoints/shufflenetv2_x1_0.h5'
    # assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)
    # model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)
elif model_name == "EfficientnetV1":
    model = efficientnet_b0(input_shape=(im_height, im_width, 3),
                               num_classes=num_classes)
    model.build((im_height, im_width, 3))
    pre_weights_path = './checkpoints/efficientnetb0.h5'
    assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)
    model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)
elif model_name == "EfficientnetV2":
    model = efficientnetv2_s(num_classes=num_classes)
    model.build((1,im_height, im_width, 3))
    pre_weights_path = './checkpoints/tf_efficientnetv2/efficientnetv2-s.h5'
    assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)
    model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

model.summary()
# ---------------------------------------------------------------------------------------------------------------------#

# 自定义学习率曲线
initial_lr = 0.01 # 学历率的初始值
def scheduler(now_epoch):
    end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
    rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
    new_lr = rate * initial_lr

    return new_lr

# 优化器
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
if model_name in ["MobilenetV3"]:
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr,momentum=0.9)

# 
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)  # 前向推理
        loss = loss_object(labels, predictions)     # 计算损失
    gradients = tape.gradient(loss, model.trainable_variables)  # 反向传播
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
    
@tf.function
def train_googlenet_step(images, labels): # 用于googlenet网络训练
    with tf.GradientTape() as tape:
        aux1, aux2, output = model(images, training=True)
        loss1 = loss_object(labels, aux1)
        loss2 = loss_object(labels, aux2)
        loss3 = loss_object(labels, output)
        loss = loss1 * 0.3 + loss2 * 0.3 + loss3
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, output)

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    
@tf.function
def test_googlenet_step(images, labels): # 用于googlenet网络验证
    _, _, output = model(images, training=False)
    t_loss = loss_object(labels, output)

    test_loss(t_loss)
    test_accuracy(labels, output)    


# ------------------------------------------------ 训练 ----------------------------- #
best_test_loss = float('inf') # 记录最小的损失
train_step_num = train_num // batch_size
val_step_num = val_num // batch_size
for epoch in range(1, epochs+1):
    train_loss.reset_states()        # clear history info
    train_accuracy.reset_states()    # clear history info
    test_loss.reset_states()         # clear history info
    test_accuracy.reset_states()     # clear history info

    t1 = time.perf_counter()
    # 训练集推理
    for index, (images, labels) in enumerate(train_dataset):
        if model_name == "Googlenet": 
            train_googlenet_step(images, labels)
        else:
            train_step(images, labels)
        if index+1 == train_step_num:
            break
    # print(time.perf_counter()-t1)
    logger.info("该epoch花费时间："+str(time.perf_counter()-t1))
    # 验证集推理
    for index, (images, labels) in enumerate(val_dataset):
        if model_name == "Googlenet": 
            test_googlenet_step(images, labels)
        else:
            test_step(images, labels)
        if index+1 == val_step_num:
            break
    # update learning rate
    optimizer.learning_rate = scheduler(epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    # print(template.format(epoch,
    #                         train_loss.result(),
    #                         train_accuracy.result() * 100,
    #                         test_loss.result(),
    #                         test_accuracy.result() * 100))
    # 记录训练信息
    logger.info(template.format(epoch,
                            train_loss.result(),
                            train_accuracy.result() * 100,
                            test_loss.result(),
                            test_accuracy.result() * 100))
    if test_loss.result() < best_test_loss: # 保存损失最小的模型
        model.save_weights(f"./save_weights/my{model_name}.h5")    
logger.info("训练结束!")
# --------------------------------------------------------------------------------------------- #

# ------------------------------------- 根据日志信息绘图 --------------------------------------- #
if log_draw_loss_suc:
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    with open(f"logs/{model_name}.log",encoding='utf-8') as file:
        lines = file.readlines()
        for i in lines:
            if "Epoch" in i:
                train_loss.append(float(i.strip().split("Loss:")[1].split(",")[0]))
                train_accuracy.append(float(i.strip().split("Accuracy:")[1].split(",")[0]))
                val_loss.append(float(i.strip().split("Test Loss:")[1].split(",")[0]))
                val_accuracy.append(float(i.strip().split("Test Accuracy:")[1]))

    # figure 1
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    # figure 2
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()
# --------------------------------------------------------------------------------------------- #