import os
import tensorflow as tf
import numpy as np
import json
from utils import mnist_reader

x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='t10k')
x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("開始訓練模型...")
history = model.fit(
    x_train, y_train_onehot,
    batch_size=128,
    epochs=20,
    validation_data=(x_test, y_test_onehot),
    verbose=1
)

test_loss, test_acc = model.evaluate(x_test, y_test_onehot, verbose=0)
print(f'Final Test accuracy: {test_acc:.4f}')

if not os.path.exists('model'):
    os.makedirs('model')

model.save('model/fashion_mnist.h5')
print("模型已保存為 fashion_mnist.h5")


def extract_model_info(model_path):
    model = tf.keras.models.load_model(model_path)

    architecture = []

    for layer in model.layers:
        layer_config = {
            'name': layer.name,
            'type': layer.__class__.__name__,
            'config': {},
            'weights': []
        }

        if isinstance(layer, tf.keras.layers.Dense):
            activation_name = layer.activation.__name__
            if activation_name == 'linear':
                activation_name = None
            layer_config['config']['activation'] = activation_name
            layer_config['weights'] = [
                f'{layer.name}/kernel:0', f'{layer.name}/bias:0']
        elif isinstance(layer, tf.keras.layers.Flatten):
            pass

        architecture.append(layer_config)

    with open('model/fashion_mnist.json', 'w') as f:
        json.dump(architecture, f, indent=2)
    print("架構已保存為 fashion_mnist.json")

    weights_dict = {}
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            weights = layer.get_weights()
            if isinstance(layer, tf.keras.layers.Dense):
                weights_dict[f'{layer.name}/kernel:0'] = weights[0]
                weights_dict[f'{layer.name}/bias:0'] = weights[1]

    np.savez('model/fashion_mnist.npz', **weights_dict)
    print("權重已保存為 fashion_mnist.npz")


extract_model_info('model/fashion_mnist.h5')

print("\n驗證提取的權重...")
loaded_weights = np.load('model/fashion_mnist.npz')
print("提取的權重檔案包含以下鍵值:")
for key in loaded_weights.files:
    print(f"  {key}: shape {loaded_weights[key].shape}")
