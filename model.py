from tensorflow.keras.applications.vgg16 import VGG16

pretrained_VGG16 = VGG16(input_shape = (512, 512, 3), include_top = False, weights='imagenet')
pretrained_VGG16.trainable = False
pretrained_VGG16.summary()
