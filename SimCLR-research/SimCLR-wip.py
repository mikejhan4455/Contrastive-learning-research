from tensorflow import keras 
from keras import layers, activations, losses, callbacks, models, applications
from tensorflow.python.keras.mixed_precision import loss_scale
import tensorflow_datasets as tfds
import tensorflow as tf

def load_dataset():
    train_data, validation_data = tfds.load('imagenette', split=[
                                            'train', 'validation'], data_dir='./imagenette', as_supervised=True)
    train_data = train_data.map(lambda i, j: tf.image.resize(i, (224, 224)))  # type: ignore
    validation_data = validation_data.map(lambda i, j: tf.image.resize(i, (224, 224)))  # type: ignore
    return train_data.repeat().batch(32), validation_data.repeat().batch(32)




class NT_Xent(tf.keras.layers.Layer):
    """ Normalized temperature-scaled CrossEntropy loss [1]
        [1] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” arXiv. 2020, Accessed: Jan. 15, 2021. [Online]. Available: https://github.com/google-research/simclr.
    """

    def __init__(self, tau=1, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.similarity = tf.keras.losses.CosineSimilarity(
            axis=-1, reduction=tf.keras.losses.Reduction.NONE)
        self.criterion = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True)

    def get_config(self):
        return {"tau": self.tau}

    def call(self, zizj):
        """ zizj is [B,N] tensor with order z_i1 z_j1 z_i2 z_j2 z_i3 z_j3 ... 
            batch_size is twice the original batch_size
        """
        batch_size = tf.shape(zizj)[0]
        mask = tf.repeat(
            tf.repeat(~tf.eye(batch_size/2, dtype=tf.bool), 2, axis=0), 2, axis=1)

        sim = -1*self.similarity(tf.expand_dims(zizj, 1),
                                 tf.expand_dims(zizj, 0))/self.tau
        sim_i_j = -1*self.similarity(zizj[0::2], zizj[1::2])/self.tau

        pos = tf.reshape(tf.repeat(sim_i_j, repeats=2), (batch_size, -1))
        neg = tf.reshape(sim[mask], (batch_size, -1))

        logits = tf.concat((pos, neg), axis=-1)
        labels = tf.one_hot(
            tf.zeros((batch_size,), dtype=tf.int32), depth=batch_size-1)

        return self.criterion(labels, logits)



def contrastive_loss(projections_1, projections_2, temperature=1):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)

        # TODO: logic problem fix here
        contrastive_labels, tf.transpose(similarities)
        

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2


def build_model():

    # input_layer = layers.Input(shape=(None, None, 3))  # type: ignore
    # # resizing = layers.Resizing(height=224, width=224)(input_layer)  # type: ignore
    
    # resnet50_imagenet_model = applications.ResNet50(
    #     include_top=True, weights='imagenet', input_shape=(224, 224, 3))(input_layer)

    # #Flatten output layer of Resnet
    # flattened = tf.keras.layers.Flatten()(resnet50_imagenet_model)

    # #Fully connected layer 1
    # fc1 = tf.keras.layers.Dense(
    #     128, activation='relu', name="AddedDense1")(flattened)

    # #Fully connected layer, output layer
    # fc2 = tf.keras.layers.Dense(
    #     12, activation='softmax', name="AddedDense2")(fc1)

    # return tf.keras.models.Model(
    #     inputs=input_layer, outputs=fc2)

    return applications.ResNet50(include_top=True)



def main():
    
    train_ds, validation_ds = load_dataset()
    
    model = build_model()
    model.summary()

    model.compile(loss=contrastive_loss, optimizer='adam')

    # parameters
    BATCH_SIZE = 32
    EPOCHS = 10


    # print(train_ds.next()[0].shape)

    model.fit(x=train_ds, epochs=EPOCHS, steps_per_epoch=32)



if __name__ == '__main__':
    main()

