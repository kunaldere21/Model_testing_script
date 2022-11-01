import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB0
import os
# from tensorflow_model_optimization.quantization.keras import vitis_quantize
h5_models_path = '/home/dell/Documents/My_wrok/oct_19/temp/' # folder where all .h5 model present 
tf_lite_models_path = '/home/dell/Documents/My_wrok/oct_19/temp/' # destination path
os.makedirs(tf_lite_models_path,exist_ok=True)

for model in os.listdir(h5_models_path):
    model_name = model.split('.')[0]+'.tflite'
    new_model= tf.keras.models.load_model(filepath=os.path.join(h5_models_path,model))
    # Tflite model conversion
    tflite_converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
    tflite_model = tflite_converter.convert()
   
    open(os.path.join(tf_lite_models_path,model_name), "wb").write(tflite_model)



# # # Quantization technique

# # quantizer = vitis_quantize.VitisQuantizer(new_model)
# # quantizer.quantize_model(calib_dataset,
# #                          input_layers=['conv2'],
# #                          bias_bit=32, 
# #                          activation_bit=32, 
#                          weight_per_channel=True)
