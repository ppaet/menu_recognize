import os
import json
import cv2
import numpy as np
import tensorflow as tf

# 加载标签映射和模型信息
def load_model_info(model_info_path):
    with open(model_info_path, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    return model_info

# 加载模型
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# 处理输入图像
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, (224, 224))  # 调整到模型输入大小
        return np.array(image) / 255.0  # 归一化处理
    else:
        raise FileNotFoundError(f"Image not found: {image_path}")

# 进行预测
def predict(image_path, model, label_to_index, max_dishes):
    # 处理图像
    input_image = process_image(image_path)
    input_image = np.expand_dims(input_image, axis=0)  # 增加批次维度

    # 进行预测
    predictions = model.predict(input_image)

    predicted_labels = []
    predicted_prices = []

    for i in range(max_dishes):
        label_pred = np.argmax(predictions[i], axis=-1)
        price_pred = predictions[max_dishes + i]

        # 处理有效的预测结果
        for label in label_pred:
            if label != 0:  # 假设0是填充标签
                predicted_labels.append(label)  # 存储标签
                predicted_prices.append(float(price_pred[0]))  # 存储价格并转换为浮动数字

    return predicted_labels, predicted_prices

# 输出预测结果
def print_predictions(predicted_labels, predicted_prices, index_to_label):
    for i, label in enumerate(predicted_labels):
        dish_name = index_to_label[label]
        price = predicted_prices[i]
        print(f"Dish {i + 1}: {dish_name}, Predicted Price: ${price:.2f}")

# 主程序
if __name__ == "__main__":
    model_info_path = './model_info.json'
    model_path = './menu_recognition_model.h5'
    
    # 加载模型信息和模型
    model_info = load_model_info(model_info_path)
    label_to_index = model_info['label_to_index']
    max_dishes = model_info['max_dishes']
    
    # 创建 index_to_label 映射
    index_to_label = {v: k for k, v in label_to_index.items()}

    # 加载模型
    model = load_model(model_path)

    # 预测的图片路径
    image_path = './image.png'  # 替换为你的图片路径

    # 进行预测
    predicted_labels, predicted_prices = predict(image_path, model, label_to_index, max_dishes)

    # 输出预测结果
    print_predictions(predicted_labels, predicted_prices, index_to_label)
