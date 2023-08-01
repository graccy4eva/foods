import tensorflow as tf
from tensorflow.python.keras.saving.saved_model import load as load_model
#model = tf.saved_model.load('C:/Users/owner/AppData/Local/Programs/Python/Python36/Myprogs/Deployment-Deep-Learning-Model-master/saved_model.pb')
#keras_model = load_model(model)
#keras_model.save('save_model.h5')
#model = tf.saved_model.load('models')
#keras_model = load_model(model)
#keras_model.save('models/save_model.h5')

file_pb = "models"
file_h5 = "models/save_model.h5"
loaded_model = tf.keras.models.load_model(file_pb)
tf.keras.models.save_model(loaded_model, file_h5)