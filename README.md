# DreamyFood
This project is completed by Bjarke Larsen, Ethan Osborne, and Anya Osborne as part of the final class project about artifacts that appreciate art. It is based on the output of the pre-trained Generative Adversarial Network (GAN) model developed within the Project 2 - WoWIconGAN. The images generated as a result of the WoWIconGAN were used as input for the present project. The idea behind the project is to have the machine create new stylized World of Warcraft icon art and “understand” it by classifying the generated images within a specific domain. We chose to use 101-Food types for this domain that functions as a desired classification vector in our Google Deep Dream model that utilizes trained Inception V3 model and Gradient Ascent. For the food classifier, we modified the harimkang / food-image-classifier as a reference and included its test mode into our project.
Example of Output
The Output includes two steps: (1) Generating dreamified images using gradient ascent; (2) Classification of the original image (forbatch) and the resulted deamified images using pre-trained 101-food classification model. FYI download images here for the GitHub repository.
Step 1: Generating dreamified images using gradient ascent 
Upload images you want to use in the forbatch folder. They can be of any size and resolution. The program will automatically rescale them. It will first run them via Deep Dream model using the Gradient Ascent, which will try to maximize the activations of specific layers for this input based on the food-trained Inception V3 model.

Step 2: Classification of the original image and the deamified images using food domain.
It saves classified images in the food_classification_saved folder, where it classifies both the original image placed in the forbatch folder and the dreamified output. 

How to Run the Code
Download this repository
Set up your environment
Import Tensorflow using pip install
Import CUDA if you would like to use a GPU device
Install cuDnn into your environment 
To avoid conflicts between packages for Tensorflow, please review the compatibility tables
Python 3+ 
Import numpy
Import matplotlip library
Set up the correct directory addresses for the outputs to be generated to using the foodifywithtest code (see lines 43-46) 
Run the foodifywithtest.py script
The output will be generated in the Abstract-To_Food folder
Note that you can you any images you want, make sure to upload them into the forbatch folder.
Rules and Constraints
To use Gradient Ascent in Google Deep Dream model, you will need to plug the pre-trained classifier model and related checkpoints into the foodifywithtest code. The pre-trained model we used is available in the models folder. It must be files with the <.hdf5> extension. You can use any other model you wish to apply that uses an Inception V3 CNN. Inception v3 is an image recognition model that has been shown to attain greater than 78.1% accuracy on the image dataset. Our 101-food model is at 82.5% of accuracy. 
Process in Creating a Good Output
To tweak the output for dreamified images, use the following parameters in the foodifywithtest code (see lines 28-40).
_target_size - Adjusts the resolutions of the dreamified images by upscaling the starting image (for example, 256=>1K, 516=>2K).
_tile_size - Sets up the size for the rolled tiling (default is 512).
_octaverangemin and _octaverangemax - Determines the amount of octaves and their scales. More ranges will cause more images of increasingly higher detail. More ranges also requires higher starting resolution.
_octavescale - It is the default scale of each octave, which is further scaled by the octave ranges. We found 2 to be nice, but DeepDream started with 1.3.
_steps_per_octave - Defines number of steps for octave.
_step_size - Defines the step size for each octave.
_save_per_step - If this number is higher than steps per octave, it will not work.
layersnames - It defines what layers it will use to maximize the activations. These layers may vary from 0 to 10. Deeper (higher) levels define high level features (for example, eyes, faces). Lower layers define more basic features like shapes.

