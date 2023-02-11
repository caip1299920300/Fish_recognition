from tensorflow.keras import layers, Model, models, Sequential

def AlexNet_v1(im_height=224, im_width=224, num_classes=1000):
    # tensorflow中tensor通道排序为NHWC
    # [None, 224,224,3]
    input_image = layers.Input(shape=(im_height,im_width, 3) ,dtype="float32")
    # [None, 227,227,3]
    x = layers.ZeroPadding2D(((1,2),(1,2)))(input_image)
    # [None, 55,55,48]
    x = layers.Conv2D(48, kernel_size=11, strides=4, activation="relu")(x)
    # [None, 27,27,48]
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)
    # [None, 13,13,128]
    x = layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(x)
    # [None, 13,13,128]
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)
    # [None, 13,13,192]
    x = layers.Conv2D(192, kernel_size=5, padding="same", activation="relu")(x)
    # [None, 13,13,192]
    x = layers.Conv2D(192, kernel_size=5, padding="same", activation="relu")(x)
    # [None, 13,13,128]
    x = layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(x)
    # [None, 6,6,128]
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)
    # [None , 6*6*128]
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    # [None , 1024]
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(num_classes)(x)
    predict = layers.Softmax()(x)

    model = models.Model(inputs=input_image, outputs=predict)
    return model

class AlexNet_v2(Model):
    def __init__(self, num_classes=1000):
        super(AlexNet_v2, self).__init__()
        self.features = Sequential([
            layers.ZeroPadding2D(((1,2),(1,2))),
            layers.Conv2D(48, kernel_size=11, strides=4, activation="relu"),
            # [None, 27,27,48]
            layers.MaxPool2D(pool_size=3, strides=2),
            # [None, 13,13,128]
            layers.Conv2D(128, kernel_size=5, padding="same", activation="relu"),
            # [None, 13,13,128]
            layers.MaxPool2D(pool_size=3, strides=2),
            # [None, 13,13,192]
            layers.Conv2D(192, kernel_size=5, padding="same", activation="relu"),
            # [None, 13,13,192]
            layers.Conv2D(192, kernel_size=5, padding="same", activation="relu"),
            # [None, 13,13,128]
            layers.Conv2D(128, kernel_size=5, padding="same", activation="relu"),
            # [None, 6,6,128]
            layers.MaxPool2D(pool_size=3, strides=2)
        ])
        
        self.flatten = layers.Flatten()
        self.classifier = Sequential([
            layers.Dropout(0.2),
            # [None , 1024]
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes),
            layers.Softmax()
        ])
    
    def call(self, inputs, **kwargs):
        x = self.features(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x