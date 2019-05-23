import matplotlib.pyplot as plt
import math
import os
import tensorflow as tf
import tensorflow_datasets as tfds


def main(train, evaluate):
    tf.enable_eager_execution()

    # Download the "Rock, Paper, Scissors" dataset with labels and split into training and test
    # @ONLINE {rps,
    # author = "Laurence Moroney",
    # title = "Rock, Paper, Scissors Dataset",
    # month = "feb",
    # year = "2019",
    # url = "http://laurencemoroney.com/rock-paper-scissors-dataset"
    # }
    dataset, metadata = tfds.load(name="rock_paper_scissors", as_supervised=True, with_info=True)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Take a single image from the dataset and plot it
    save_plotted_image(train_dataset)

    train_dataset = train_dataset.map(normalize)
    test_dataset = test_dataset.map(normalize)
    num_train_examples = metadata.splits['train'].num_examples
    num_test_examples = metadata.splits['test'].num_examples

    # Define and compile the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(64, (3, 3)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(64, (3, 3)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(64, (3, 3)),
        tf.keras.layers.MaxPool2D(4, 4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    # Train the model and save it via a callback function as checkpoints.
    BATCH_SIZE = 64
    train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
    test_dataset = test_dataset.repeat().shuffle(num_test_examples).batch(BATCH_SIZE)

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)

    # Run `tensorboard --logdir tensorboard/` after training
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='tensorboard',
                                                          histogram_freq=0,
                                                          write_graph=True,
                                                          write_images=True)

    if train:
        model.fit(train_dataset,
                  epochs=1,
                  steps_per_epoch=math.floor(num_train_examples/BATCH_SIZE),
                  callbacks=[checkpoint_callback, tensorboard_callback])
    if evaluate:
        model.load_weights(checkpoint_path)

        model.evaluate(test_dataset, steps=math.floor(num_test_examples/BATCH_SIZE))

        layer_outputs = [layer.output for layer in model.layers]

        activation_model = tf.keras.models.Model(inputs=model.input,
                                                 outputs=layer_outputs)

        first_image = test_dataset.take(1)

        activations = activation_model.predict(first_image,
                                               steps=math.floor(num_test_examples/BATCH_SIZE))

        plt.figure(figsize=(20, 16))
        for num in [0, 1, 2, 3, 4, 5, 6, 7]:

            layer = activations[num]
            print(layer.shape)

            for i in range(1, 11):
                plt.subplot(8, 10, 10*num+i)
                plt.imshow(layer[0, :, :, i], cmap=plt.cm.binary)

        plt.savefig("layer_overview.png".format(num))


def save_plotted_image(train_dataset):
    for image, label in train_dataset.take(1):
        break
    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.grid(False)
    plt.xlabel("This is a {}".format(label))
    plt.savefig("test_image.png")


# Normalize the datasets from [0, 255] -> [0, 1]
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


if __name__ == "__main__":
    main(train=False, evaluate=True)
