import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Is GPU built with CUDA support:", tf.test.is_built_with_cuda())
print("Is GPU available:", tf.config.list_physical_devices('GPU'))