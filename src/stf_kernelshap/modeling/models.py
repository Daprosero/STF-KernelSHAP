import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Flatten,
    Input,
    LayerNormalization,
    Permute,
    Reshape,
    SeparableConv2D,
    SpatialDropout2D,
)
from tensorflow.keras.models import Model
from keras_nlp.layers import TransformerEncoder
from stf_kernelshap.utils import set_seed
from stf_kernelshap.modeling.layers import InspectableTransformerEncoder, inception_block, RenyiEntropyLayer


def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
             seed=True,num_seed=42):
    """ 
    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    if seed:
      set_seed(seed=num_seed)
    
    input1   = Input(shape = (Chans, Samples))

    input1 = Reshape((Chans, Samples, 1))(input1)

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   name='Conv2D_1',
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   name='Depth_wise_Conv2D_1',
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   name='Separable_Conv2D_1',
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'output', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'out_activation')(dense)
    
    return Model(inputs=input1, outputs=softmax)

# need these for ShallowConvNet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))

def ShallowConvNet(
    nb_classes,
    Chans=64,
    Samples=128,
    dropoutRate=0.5,
    n_filters=40,
    kernel_length=13,
    pool_size=35,
    pool_stride=7,
    use_bias_spatial=True,
    conv_max_norm=2.0,
    norm_rate=0.5,
    input_mode="2d",   # "2d" -> (Chans, Samples), "4d" -> (Chans, Samples, 1)
    seed=True,
    num_seed=42,
):
    """
    Versión flexible de ShallowConvNet para sintonizar con los datos.

    Parámetros
    ----------
    nb_classes : int
        Número de clases.
    Chans : int
        Número de canales.
    Samples : int
        Número de muestras temporales.
    dropoutRate : float
        Tasa de dropout.
    n_filters : int
        Número de filtros de las convoluciones iniciales.
    kernel_length : int
        Longitud del kernel temporal.
    pool_size : int
        Tamaño del average pooling sobre el eje temporal.
    pool_stride : int
        Stride del average pooling sobre el eje temporal.
    use_bias_spatial : bool
        Si la convolución espacial usa bias.
    conv_max_norm : float
        Restricción max_norm para convoluciones.
    norm_rate : float
        Restricción max_norm para la capa densa final.
    input_mode : str
        "2d"  -> entrada (Chans, Samples)
        "4d"  -> entrada (Chans, Samples, 1)
    """

    if input_mode not in ["2d", "4d"]:
        raise ValueError("input_mode debe ser '2d' o '4d'")

    if input_mode == "2d":
        input_main = Input(shape=(Chans, Samples))
        x = Reshape((Chans, Samples, 1))(input_main)
    else:
        input_main = Input(shape=(Chans, Samples, 1))
        x = input_main

    if seed:
      set_seed(seed=num_seed)

    # Conv temporal
    x = Conv2D(
        filters=n_filters,
        kernel_size=(1, kernel_length),
        name="Conv2D_1",
        kernel_constraint=max_norm(conv_max_norm, axis=(0, 1, 2)),
        use_bias=True,
        padding="valid",
    )(x)

    # Conv espacial
    x = Conv2D(
        filters=n_filters,
        kernel_size=(Chans, 1),
        use_bias=use_bias_spatial,
        name="Conv2D_2",
        kernel_constraint=max_norm(conv_max_norm, axis=(0, 1, 2)),
        padding="valid",
    )(x)

    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = Activation(square)(x)
    x = AveragePooling2D(pool_size=(1, pool_size), strides=(1, pool_stride))(x)
    x = Activation(log)(x)
    x = Dropout(dropoutRate)(x)

    x = Flatten()(x)
    x = Dense(
        nb_classes,
        kernel_constraint=max_norm(norm_rate),
        name="output",
    )(x)

    softmax = Activation("softmax", name="out_activation")(x)

    return Model(inputs=input_main, outputs=softmax, name="ShallowConvNet")


def TGARNet(
    nb_classes=2,
    Chans=19,
    Samples=512,
    norm_rate=0.25,
    alpha=2,
    num_heads=3,
    intermediate_dim=128,
    filters=3,
    dropoutRate=0.3,
    kernel_sigma=1.0,
    seed=True,
    num_seed=42,
):
    if seed:
      set_seed(seed=num_seed)
    input1 = Input(shape=(Chans, Samples))

    # 1) Reorganize data for Transformer (Samples, Chans)
    x = Permute((2, 1))(input1)

    # 2) Normalización antes del Transformer
    x = LayerNormalization()(x)

    # 3) Transformer
    transformer_encoder = InspectableTransformerEncoder(
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
        name="transformer_encoder"
    )
    x = transformer_encoder(x)

    # 4) Normalización después del Transformer
    x = LayerNormalization()(x)

    # 5) Restore original shape (Chans, Samples, 1)
    x = Permute((2, 1))(x)
    x = Reshape((Chans, Samples, 1))(x)

    # 6) Single kernel block
    kernel_out = inception_block(x, kernel_sigma)

    # 7) Single-kernel Rényi entropy
    kernel_out_T = Permute((3, 1, 2))(kernel_out)#TransposeLayer()(kernel_out)   # debe quedar (B, 1, C, C) o equivalente
    kernel_entropy = RenyiEntropyLayer(alpha=alpha,normalize_by_logc=True,name="kernel_entropy")(kernel_out_T)
    # 8) Extra convolutional stack
    final_conv = Conv2D(filters,kernel_size=3,padding='same',activation='selu',name='Conv2D_1')(kernel_out)
    final_conv = BatchNormalization()(final_conv)
    final_conv = Conv2D(filters,kernel_size=3,padding='same',activation='selu',name='Conv2D_2')(final_conv)
    final_conv = BatchNormalization()(final_conv)

    flat = Flatten()(final_conv)
    drop = Dropout(dropoutRate)(flat)
    softmax = Dense(nb_classes,'softmax',name='out_activation',kernel_constraint=max_norm(norm_rate))(drop)
    model = Model(inputs=input1,outputs={'out_activation': softmax,'kernel_entropy': kernel_entropy})
    return model
