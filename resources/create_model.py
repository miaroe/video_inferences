import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, TimeDistributed, LSTM
from tensorflow.keras.models import Model


def create_model(instance_size, num_stations, stateful):

    #base_model = tf.keras.applications.MobileNetV3Small(input_shape=instance_size, include_top=False, weights='imagenet',
    #                                                    include_preprocessing=True, minimalistic=True, dropout_rate=0.3)

    base_model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=instance_size, pooling=None)

    #for layer in base_model.layers[:-11]:
    #    layer.trainable = False


    # Make sure the correct layers are frozen
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name, layer.trainable)

    # Create the input layer for the sequence of images
    sequence_input = Input(shape=(None, *instance_size), batch_size=1)  # (B, T, H, W, C)

    # Apply the CNN base model to each image in the sequence
    x = TimeDistributed(base_model)(sequence_input)  # (B, T, H', W', C')

    # Apply Global Average Pooling to each frame in the sequence
    x = TimeDistributed(tf.keras.layers.GlobalMaxPooling2D())(x)  # (B, T, C')

    # Create an LSTM layer
    x = LSTM(32, return_sequences=True, stateful=stateful)(x)  # (B, T, lstm_output_dim)

    x = LSTM(32, return_sequences=False, stateful=stateful)(x)  # (B, lstm_output_dim)

    # Create a dense layer
    # x = Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)  # (B, dense_output_dim)
    x = Dense(32, activation='relu')(x)  # (B, dense_output_dim)

    # Create a dropout layer
    x = Dropout(0.5)(x)  # (B, dense_output_dim)

    # Create the output layer for classification
    output = Dense(num_stations, activation='softmax')(x)  # (B, num_classes)

    # Create the combined model
    model = Model(inputs=sequence_input, outputs=output)

    return model
