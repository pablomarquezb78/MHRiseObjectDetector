# MHRiseObjectDetector
MHObjectDetector is a program which takes a TFLite trained model on videogame Monster hunter rise, and makes an inference of the model on a gameplay video, it is prepared to detect health,stamina and sharp bar, felyne and canyne comrade, player, map and monster icons, concretely with the icons, this program makes an image matching between icons detected by the model and a folder with all monster icons, this mixed with a monster info JSON give us information about the monster matched with each detection, this program does this same thing with the map, also using a map JSON and a folder with map images.

On this upload I include an already trained TFLite model, trained with 200 labeled images, and 5 videos to try the inference, one for each in-game map. It is important to say that this program is based on Monster hunter Rise base game, so it only includes the 5 base game maps and no special maps.
## Step 1: Installation
I will explain the way I download this project and execute it, first of all, I use PyCharm Edu with Anaconda, so first you will have to install the latest version of [PyCharm](https://www.jetbrains.com/edu-products/download/other-PCE.html). Then install [Anaconda](https://www.anaconda.com/products/distribution) just clicking download button.

When the download finishes, open the downloaded .exe and go through the installation program using the default options.
## Step 2: Set Up Virtual Environment and Directory
Now first of all you will need to put the downloaded folder 'tflite1' where you please, but I recommend to put it in C: drive because the .py is configured to that path, now open "Anaconda Command Prompt" and move to the choosed directory for the project, on our case:

```
cd C:\tflite1
```
Next create a python 3.9 virtual enviroment:
```
conda create --name tflite1-env python=3.9
```
Enter "y" when it asks if you want to proceed. Activate the environment and install the required packages by issuing the commands below. We'll install TensorFlow, OpenCV, and a downgraded version of protobuf:
```
conda activate tflite1-env
pip install tensorflow opencv-python protobuf==3.20.*
pip install Scipy
```

*If you use another path for 'tflite1' you will have to change .py file and update the variables on function 'run_object_detection' on MHObjectDetector.py as it is commented there.*

## Optional Step: Change TFLite model:
Our project already includes a TFLite trained model, but if you want to use another one, you just need to replace the 'TFLite_model' folder on our 'tflite1' folder with your trained model, I used this google colab to create my model: https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb#scrollTo=GSJ2wgGCixy2, you can either create your model in other ways but you will probably need to make some changes on the project configuration to make it work.

## Step 4: Run the Project
Now you just need to run PyCharm open the 'tflite1' folder, make sure to **use anaconda python interpreter** by going to File, Settings, project and then python interpreter, once you have set up the interpreter you can run it! You will need to enter on the textbox of the opened window the name of the video which you would like to make the inference, on our folder I include 5 videos v1,v2..v5.mp4, but if you want, you can use your own .mp4 you just need to introduce them on the 'tflite1' folder and then enter their name (whithout .mp4) on the program.
