import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 加载图像和标签
def load_data(image_dir, annotation_file):
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    images = []
    labels = []
    prices = []

    for img_name, label_info_list in annotations.items():
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, (224, 224))
            images.append(image)

            img_labels = [item['name'] for item in label_info_list]
            img_prices = [float(item['price']) for item in label_info_list]

            labels.append(img_labels)
            prices.append(img_prices)

    return np.array(images), labels, prices

# 数据路径
image_dir = './images'  
annotation_file = './annotations.json'  

# 加载数据
X, y_labels, y_prices = load_data(image_dir, annotation_file)

# 将所有标签转换为索引编码
label_set = list({label for sublist in y_labels for label in sublist})
label_to_index = {label: i for i, label in enumerate(label_set)}

y_labels_encoded = [[label_to_index[label] for label in label_list] for label_list in y_labels]

max_dishes = max(len(labels) for labels in y_labels_encoded)

# 使用有效值填充标签和价格
padding_value = len(label_set)  # 将填充值设置为一个有效的标签索引

y_labels_padded = [np.pad(labels, (0, max_dishes - len(labels)), 'constant', constant_values=padding_value) for labels in y_labels_encoded]
y_prices_padded = [np.pad(prices, (0, max_dishes - len(prices)), 'constant', constant_values=-1.0) for prices in y_prices]

X = X / 255.0  # 图像数据归一化到0-1范围
y_labels_padded = np.array(y_labels_padded)
y_prices_padded = np.array(y_prices_padded)

# 分割数据集
X_train, X_val, y_labels_train, y_labels_val, y_prices_train, y_prices_val = train_test_split(
    X, y_labels_padded, y_prices_padded, test_size=0.2, random_state=42
)

# 构建模型
input_layer = Input(shape=(224, 224, 3))
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_layer)
x = base_model.output
x = Flatten()(x)
x = Dropout(0.5)(x)

# 多任务输出
name_outputs = []
for i in range(max_dishes):
    name_output = Dense(len(label_set) + 1, activation='softmax', name=f'name_output_{i}')(x)  # 输出标签的数量+1（包括填充）
    name_outputs.append(name_output)

price_outputs = []
for i in range(max_dishes):
    price_output = Dense(1, activation='linear', name=f'price_output_{i}')(x)
    price_outputs.append(price_output)

model = Model(inputs=input_layer, outputs=name_outputs + price_outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # 降低学习率
    loss=['sparse_categorical_crossentropy'] * max_dishes + ['mean_squared_error'] * max_dishes,
    metrics=['accuracy'] * max_dishes + ['mae'] * max_dishes
)

model.summary()

# 训练模型
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 生成样本权重
sample_weight = []
for i in range(max_dishes):
    weight = np.where(y_labels_train[:, i] == padding_value, 0.0, 1.0).astype(float)  # 将填充标签位置权重设置为 0
    sample_weight.append(weight)

# 对于价格输出，创建相应的权重
price_sample_weight = [np.ones_like(y_prices_train[:, i]) for i in range(max_dishes)]

# 合并标签
train_labels = [y_labels_train[:, i] for i in range(max_dishes)]  # 每个菜品名称的标签
train_prices = [y_prices_train[:, i] for i in range(max_dishes)]    # 每个菜品的价格

# 合并所有标签
y_combined = train_labels + train_prices

# 验证集标签
validation_labels = [y_labels_val[:, i] for i in range(max_dishes)]
validation_prices = [y_prices_val[:, i] for i in range(max_dishes)]

# 训练模型
history = model.fit(
    X_train,
    y_combined,
    sample_weight=sample_weight + price_sample_weight,
    validation_data=(X_val, validation_labels + validation_prices),
    epochs=50,  # 增加训练轮数
    batch_size=16,
    callbacks=[early_stopping]
)

# 保存 model_info 和模型
with open('./model_info.json', 'w', encoding='utf-8') as f:
    model_info = {
        'label_to_index': label_to_index,
        'max_dishes': max_dishes
    }
    json.dump(model_info, f, ensure_ascii=False)

# 保存模型
model.save('./menu_recognition_model.h5')

# 绘制训练过程中的准确率和损失
train_acc = np.mean([history.history[f'name_output_{i}_accuracy'] for i in range(max_dishes)], axis=0)
val_acc = np.mean([history.history[f'val_name_output_{i}_accuracy'] for i in range(max_dishes)], axis=0)

train_loss = np.mean([history.history[f'price_output_{i}_loss'] for i in range(max_dishes)], axis=0)
val_loss = np.mean([history.history[f'val_price_output_{i}_loss'] for i in range(max_dishes)], axis=0)

epochs = range(1, len(train_acc) + 1)

# 绘制准确率曲线
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc, '-r*', label='Training Accuracy')
plt.plot(epochs, val_acc, '-b*', label='Validation Accuracy')

# 绘制损失曲线
plt.plot(epochs, train_loss, '-go', label='Training Loss')
plt.plot(epochs, val_loss, '-mo', label='Validation Loss')

plt.title('Training and Validation Accuracy and Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()

# 保存图表为文件
plt.savefig('./training_plot.png')  # 指定文件名和格式，如 PNG
plt.show()
