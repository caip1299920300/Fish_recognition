from tensorflow.keras import layers, Model, Sequential

# 初始化Conv权重
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}
# 初始化Dense权重
DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

# 全连接分类网络
def VGG(feature, im_height=224, im_width=224, num_classes=1000):
    # tensorflow中的tensor通道排序是NHWC
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = feature(input_image)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu',
                     kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu',
                     kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)
    x = layers.Dense(num_classes,
                     kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)
    output = layers.Softmax()(x)
    model = Model(inputs=input_image, outputs=output)
    return model


# 特征提取网络
def make_feature(cfg):
    feature_layers = []
    for v in cfg:
        if v == "M": # 池化层
            feature_layers.append(layers.MaxPool2D(pool_size=2, strides=2))
        else:        # 卷积层
            conv2d = layers.Conv2D(v, kernel_size=3, padding="SAME", activation="relu",
                                   kernel_initializer=CONV_KERNEL_INITIALIZER)
            feature_layers.append(conv2d)
    return Sequential(feature_layers, name="feature")


# 网络配置参数
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(model_name="vgg16", im_height=224, im_width=224, num_classes=1000):
    assert model_name in cfgs.keys(), "not support model {}".format(model_name)
    cfg = cfgs[model_name]
    model = VGG(make_feature(cfg), im_height=im_height, im_width=im_width, num_classes=num_classes)
    return model