from keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, UpSampling2D, concatenate, Flatten, Dropout, Dense
import keras.backend as k
from keras.layers import Input
from keras.models import Model
from keras import layers
from keras_applications import correct_pad


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MyMobileNetV2(input_shape):
    alpha = 1
    img_input = Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)),
                             name='Conv1_pad')(img_input)
    x = layers.Conv2D(32,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      name='Conv1')(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = layers.ReLU(6., name='Conv1_relu')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=9)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=2,
                            expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=11)
    y = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=12)

    md = Model(img_input, y)

    return md


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    in_channels = k.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand_BN')(x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(k, x, 3),
                                 name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def rank_model():
    img = Input(shape=(64, 64, 3), name='input')
    x = _inverted_res_block(img, filters=4, alpha=1, stride=1,
                            expansion=6, block_id=40)
    x = _inverted_res_block(x, filters=8, alpha=1, stride=2,
                            expansion=6, block_id=41)
    x = _inverted_res_block(x, filters=8, alpha=1, stride=1,
                            expansion=6, block_id=42)
    x = _inverted_res_block(x, filters=12, alpha=1, stride=2,
                            expansion=6, block_id=45)
    x = _inverted_res_block(x, filters=12, alpha=1, stride=1,
                            expansion=6, block_id=46)
    x = _inverted_res_block(x, filters=16, alpha=1, stride=2,
                            expansion=6, block_id=47)
    x = _inverted_res_block(x, filters=16, alpha=1, stride=1,
                            expansion=6, block_id=48)
    x = _inverted_res_block(x, filters=32, alpha=1, stride=2,
                            expansion=6, block_id=49)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(11, activation='softmax', name='output')(x)

    md = Model(img, y)
    md.summary()
    md.compile(optimizer='adam', loss='categorical_crossentropy')
    return md


def mini_yolo(num_classes, num_anchors, input_shape=(None, None, 3)):
    mbnet = MyMobileNetV2(input_shape)

    x, y1 = make_last_layers(mbnet.get_layer('block_12_add').output, 96, num_anchors * (num_classes + 5))

    x = point_conv(x, 64)
    x = UpSampling2D()(x)
    x = concatenate([x, mbnet.get_layer('block_9_add').output])
    x, y2 = make_last_layers(x, 64, num_anchors * (num_classes + 5))

    model = Model(mbnet.input, [y1, y2])
    model.summary()
    return model


def depth_conv(inputs, num_filters):
    x = DepthwiseConv2D(kernel_size=3,
                        activation=None,
                        use_bias=False,
                        padding='same')(inputs)

    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU()(x)

    # x = Conv2D(num_filters,
    #            kernel_size=1,
    #            padding='same',
    #            use_bias=False,
    #            activation=None)(x)
    # x = BatchNormalization(
    #     epsilon=1e-3, momentum=0.999)(x)
    # x = ReLU()(x)

    return x


def point_conv(inputs, num_filters):
    x = Conv2D(num_filters,
               kernel_size=1,
               padding='same',
               use_bias=False,
               activation=None)(inputs)
    x = BatchNormalization(
        epsilon=1e-3, momentum=0.999)(x)
    x = ReLU()(x)
    return x


def make_last_layers(inputs, num_filters, out_filters):
    x = point_conv(inputs, num_filters)
    x = depth_conv(x, num_filters * 2)
    x = point_conv(x, num_filters)
    x = depth_conv(x, num_filters * 2)
    x = point_conv(x, num_filters)
    y = depth_conv(x, num_filters * 2)
    y = Conv2D(out_filters, kernel_size=(1, 1))(y)
    return x, y
