from models import generator

generator = keras.models.load_model("C:/Notebooks/GANModels/avatar_generator_50.h5")

noise = tf.random.normal([1, SEED_SIZE])
generated_image = generator(noise, training=False)
print("Generated image shape: ", generated_image.shape)

img = keras.preprocessing.image.array_to_img(generated_image)
img.save("C:/Notebooks/GANModels/images/generated_img.png" % (epoch, i))