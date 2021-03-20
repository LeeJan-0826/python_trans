import os
# 分别定义目录
train_horse_dir = os.path.join('E:\\Python_Project\\human_horse\\horses')
train_human_dir = os.path.join('E:\\Python_Project\\human_horse\\humans')


def get_path(example_dir):
    pic_path=[]
    for path in os.listdir(example_dir):
        real_path = os.path.join(example_dir, path)
        pic_path.append(real_path)
    return pic_path

example_dir = os.path.join('E:\\Python_Project\\human_horse\\example')
example_name = get_path(example_dir)
print(example_name)
import numpy as np
from tensorflow.keras.preprocessing import image
for filename in example_name:
    # predicting images
    img = image.load_img(filename)
    print(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
# 现在，让我们看看 "马 "和 "人 "训练目录中的文件名是什么样的。
train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])
train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])
# 我们来看看目录中马和人的图片总数。
# print('total training horse images:', len(os.listdir(train_horse_dir)))
# print('total training human images:', len(os.listdir(train_human_dir)))

# 该程序作用为统一数据集图片大小
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 创建两个数据生成器，一个训练一个验证，指定scaling范围0~1
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)
# 指定训练数据文件夹
train_generator = train_datagen.flow_from_directory(
        'E:\\Python_Project\\human_horse\\horse-or-human',  # 训练数据所在文件夹
        target_size=(150, 150),  # 指定输出尺寸
        batch_size=32, #128
        class_mode='binary')  # 指定二分类
# 指定测试数据文件夹
validation_generator = validation_datagen.flow_from_directory(
        'E:\\Python_Project\\human_horse\\validation-horse-or-human',  # 测试数据所在文件夹
        target_size=(150, 150),  # 指定输出尺寸
        batch_size=32, #128
        class_mode='binary')  # 指定二分类

print('打印generator的形状')
print(train_generator)
print(validation_generator)

# ------------------构造模型-------------------------------

import tensorflow as tf
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

# ----------训练模型---------------------
# ------------------------------------------------
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),  # 我们将使用学习率为0.001的rmsprop优化器。在这种情况下，使用RMSprop优化算法比随机梯度下降(SGD)更可取，因为RMSprop可以为我们自动调整学习率
              metrics=['acc'])
history = model.fit(
      train_generator,
      steps_per_epoch=8,
      epochs=15,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8
)

# 使用模型进行实际预测。这段代码将允许你从文件系统中选择1个或多个文件，然后它将上传它们，并通过模型判断给出图像是马还是人。
import numpy as np
from tensorflow.keras.preprocessing import image
# import matplotlib.image as mpimg

import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片


for filename in example_name:
    # predicting images
    img = image.load_img(filename, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    lena = mpimg.imread(filename)  # 读取和代码处于同一目录下的 lena.png
    # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
    # lena.shape  # (512, 512, 3)
    if classes[0] > 0.5:
        plt.xlabel('human')  # 不显示坐标轴
    else:
        plt.xlabel('horse')  # 不显示坐标轴
    plt.imshow(lena)  # 显示图片
    plt.show()
    print(classes[0])


