import os
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist

import matplotlib as mpl
import matplotlib.image as mpimg 

import IPython.display as display
import PIL.Image

from tensorflow.keras.preprocessing import image

from model import Inception_v3 as classificationModel #this requires model to be in the same folder.


#Use these lines if you get cudnn errors! They work for me (Bjarke) but no one else
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)




#PARAMETERS
_target_size = (256,256) #adjusts the resolutions of the dreamified images by upscaling the starting image, 256 -> 1K, 512 -> 2K
_tile_size = 512 #tile size for the rolled tiling. Default=512

_octaverangemin = -2 #These two determine the amount of octaves and their scales (combined with _octavescale) More ranges will cause more images of increasingly higher detail (More ranges also requires higher starting resolution!!)
_octaverangemax = 3

_octavescale = 2 #The default scale of each octave, which is further scaled by the octave ranges. I found 2 to be nice, but DeepDream started at 1.3.

_steps_per_octave = 100
_step_size = 0.01
_save_per_step = 100  #if this number is higher than steps per octave, it won't work atm.

layernames = ['mixed2', 'mixed3'] # Choose the layers to maximize the activations of (Layer 0-10 apparently. deeper (higher) is higher level features (eyes, faces), earlier numbers are more basic features (shapes) )


#CHANGE DIRECTORIES HERE
directory = 'Abstract_To_Food/forbatch/'
rawGA_output_savelocation = "Abstract_To_Food/dreamedimages/"
prediction_savelocation="Abstract_To_Food/food_classification_saved/"
textfile_location = "Abstract_To_Food/classificationresults.txt"


#Setup of global variables
iteration = 0
imagenames = []
images = []


#Finding all the images and putting them in lists
for filename in os.listdir(directory):
  if filename.endswith('.png'):
    print(filename)

    fileimg = tf.keras.preprocessing.image.load_img(
    directory+filename, grayscale=False, color_mode='rgb', target_size=_target_size,
    interpolation='nearest')
    images.append(fileimg)
    imagenames.append(filename.replace('.png', ''))

    continue
  else:
    continue


original_imgs = []

for img in images:
  original_imgs.append(np.array(img))
#show(original_img)

#original_img = original_imgs[0] #here for things that reference the size and shape of an image.



#PRETRAINED Inception Model
base_model = tf.keras.models.load_model('models/food_classifier_model_202103061933.hdf5')
base_model.load_weights("models/checkpoint/food_classifier_checkpoint_202103061933.hdf5")

#base_model.summary()

layers = [base_model.get_layer(name).output for name in layernames]

# Create the feature extraction model
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)




# Normalize an image
def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)

# Display an image
def show(img):
  display.display(PIL.Image.fromarray(np.array(img)))
  PIL.Image.fromarray(np.array(img)).show()



def save(img, octave, step):
	image = PIL.Image.fromarray(np.array(img))
	image.save(rawGA_output_savelocation + "d_" + imagenames[iteration] + "_" + str(abs(_octaverangemin)+octave)+"_"+str(step)+".png")



def calc_loss(img, model):
  # Pass forward the image through the model to retrieve the activations.
  # Converts the image into a batch of size 1.
  img_batch = tf.expand_dims(img, axis=0)
  layer_activations = model(img_batch)
  if len(layer_activations) == 1:
    layer_activations = [layer_activations]

  losses = []
  for act in layer_activations:
    loss = tf.math.reduce_mean(act)
    losses.append(loss)

  return  tf.reduce_sum(losses)


#This is the core of the Gradient Ascent
class DeepDream(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),)
  )
  def __call__(self, img, steps, step_size):
      print("Tracing")
      loss = tf.constant(0.0)
      for n in tf.range(steps):
        with tf.GradientTape() as tape:
          # This needs gradients relative to `img`
          # `GradientTape` only watches `tf.Variable`s by default
          tape.watch(img)
          loss = calc_loss(img, self.model)

        # Calculate the gradient of the loss with respect to the pixels of the input image.
        gradients = tape.gradient(loss, img)

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8 

        # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
        # You can update the image by directly adding the gradients (because they're the same shape!)
        img = img + gradients*step_size
        img = tf.clip_by_value(img, -1, 1)

      return loss, img


deepdream = DeepDream(dream_model)


def random_roll(img, maxroll):
  # Randomly shift the image to avoid tiled boundaries.
  shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
  img_rolled = tf.roll(img, shift=shift, axis=[0,1])
  return shift, img_rolled

shift, img_rolled = random_roll(np.array(original_imgs[iteration]), _tile_size)
#show(img_rolled)

class TiledGradients(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),)
  )
  def __call__(self, img, tile_size=_tile_size):
    shift, img_rolled = random_roll(img, tile_size)

    # Initialize the image gradients to zero.
    gradients = tf.zeros_like(img_rolled)

    # Skip the last tile, unless there's only one tile.
    xs = tf.range(0, img_rolled.shape[0], tile_size)[:-1]
    if not tf.cast(len(xs), bool):
      xs = tf.constant([0])
    ys = tf.range(0, img_rolled.shape[1], tile_size)[:-1]
    if not tf.cast(len(ys), bool):
      ys = tf.constant([0])

    for x in xs:
      for y in ys:
        # Calculate the gradients for this tile.
        with tf.GradientTape() as tape:
          # This needs gradients relative to `img_rolled`.
          # `GradientTape` only watches `tf.Variable`s by default.
          tape.watch(img_rolled)

          # Extract a tile out of the image.
          img_tile = img_rolled[x:x+tile_size, y:y+tile_size]
          loss = calc_loss(img_tile, self.model)

        # Update the image gradients for this tile.
        gradients = gradients + tape.gradient(loss, img_rolled)

    # Undo the random shift applied to the image and its gradients.
    gradients = tf.roll(gradients, shift=-shift, axis=[0,1])

    # Normalize the gradients.
    gradients /= tf.math.reduce_std(gradients) + 1e-8 

    return gradients

get_tiled_gradients = TiledGradients(dream_model)


def run_deep_dream_with_octaves(img, steps_per_octave=_steps_per_octave, step_size=_step_size, 
                                octaves=range(_octaverangemin,_octaverangemax), octave_scale=_octavescale):
  base_shape = tf.shape(img)
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = tf.keras.applications.inception_v3.preprocess_input(img)

  initial_shape = img.shape[:-1]
  img = tf.image.resize(img, initial_shape)
  for octave in octaves:
    # Scale the image based on the octave
    new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32)*(octave_scale**octave)
    img = tf.image.resize(img, tf.cast(new_size, tf.int32))

    for step in range(steps_per_octave):
      gradients = get_tiled_gradients(img)
      img = img + gradients*step_size
      img = tf.clip_by_value(img, -1, 1)

      if step % _save_per_step == 0:
        display.clear_output(wait=True)
        #show(deprocess(img))
        save(deprocess(img), octave, step)
        print ("Octave {}, Step {}".format(octave, step))

  result = deprocess(img)
  return result







#Loading the Classifier Model
class_list = "food-101/meta/classes.txt"
with open(class_list, "r") as txt:
    label = [read.strip() for read in txt.readlines()]

classmodel = classificationModel(
    class_list=label,
    img_width=256,    #these values came from the model. I don't fully know if they impact anything.
    img_height=256,
    batch_size=16,
)

print("Classes :", len(label))
classmodel.load()




# RUNNING THE ACTUAL CODE HERE:




for img in original_imgs:
  img = run_deep_dream_with_octaves(img=original_imgs[iteration], step_size=0.01)
  display.clear_output(wait=True)
  base_shape = tf.shape(img)[:-1]
  img = tf.image.resize(img, base_shape)
  img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)
  #show(img)
  print(" --- Image " + str(iteration) + " done!")
  print("Classification Time:")
  prediction_original = classmodel.prediction(img_path=directory+imagenames[iteration]+".png", imagename = imagenames[iteration] + "_original", savelocation = prediction_savelocation)
  prediction_foodified = classmodel.prediction(img_path=rawGA_output_savelocation + "d_" + imagenames[iteration] + "_" + str(_octaverangemax+1) + "_0.png", imagename = imagenames[iteration] + "_foodified", savelocation = prediction_savelocation)

  print("Predictions:  Original: " + prediction_original + " --  New: " + prediction_foodified)
  classificationresults = open(textfile_location, "a")
  classificationresults.write(prediction_original + "," + prediction_foodified + "\n")
  classificationresults.close()

  iteration = iteration + 1

