import tkinter as tk
import cv2
import numpy as np
import json
import os
from PIL import Image, ImageTk
from scipy import signal
from threading import Thread

icons_path = './monsterIcons/'
maps_path = './maps/'


def bigger(i1, i2, i3):
    return i1 >= i2 and i1 >= i3


# Function to crop icon excess margins on Corners
def cropCorners(image, threshold, OFFSET):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Iterating from corner right to left until we find Icon edge and then crop
    j = image.shape[1] - 1
    while not(gray[image.shape[0] - 1, j] <= (threshold + OFFSET) and gray[image.shape[0] - 1, j] >= (
            threshold - OFFSET)) and j > 0:
        j -= 1
    if j != 0:
        image = image[:, :j]

    # Iterating from corner left to right until we find Icon edge and then crop
    j = 0
    while not(gray[image.shape[0] - 1, j] <= (threshold + OFFSET) and gray[image.shape[0] - 1, j] >= (
            threshold - OFFSET)) and j < image.shape[1] - 1:
        j += 1
    if j != image.shape[1] - 1:
        image = image[:, j:]

    return image


# Function to crop icon excess margins
def cropEdges(image, threshold, OFFSET):
    # Convert image to gray
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Keep Image center (mRow,mCol)
    mRow = int(image.shape[0] / 2)
    mCol = int(image.shape[1] / 2)

    # Iterating down from upper-middle until we find Icon edge and then crop
    i = 0
    while not(gray[i, mCol] <= (threshold + OFFSET) and gray[i, mCol] >= (threshold - OFFSET)) and i < image.shape[0] - 1:
        i += 1
    if i != image.shape[0] - 1:
        image = image[i:, :]

    # Iterating up from bottom-middle until we find Icon edge and then crop
    i = image.shape[0] - 1
    while not(gray[i, mCol] <= (threshold + OFFSET) and gray[i, mCol] >= (threshold - OFFSET)) and i > 0:
        i -= 1
    if i != 0:
        image = image[:i, :]

    # Iterating right from left-middle until we find Icon edge and then crop
    j = 0
    while not(gray[mRow, j] <= (threshold + OFFSET) and gray[mRow, j] >= (threshold - OFFSET)) and j < image.shape[1] - 1:
        j += 1
    if j != image.shape[1] - 1:
        image = image[:, j:]

    # Iterating left from right-middle until we find Icon edge and then crop
    j = image.shape[1] - 1
    while not(gray[mRow, j] <= (threshold + OFFSET) and gray[mRow, j] >= (threshold - OFFSET)) and j > 0:
        j -= 1
    if j != 0:
        image = image[:, :j]

    return image


def gaussian_smoothing(image, sigma, w_kernel):
    # Define 1D kernel
    s = sigma
    w = w_kernel
    kernel_1D = np.array([1 / (s * np.sqrt(2 * np.pi)) * np.exp(-(z * z) / (2 * s * s)) for z in range(-w, w + 1)])

    # Apply distributive property of convolution
    vertical_kernel = kernel_1D.reshape(2 * w + 1, 1)
    horizontal_kernel = kernel_1D.reshape(1, 2 * w + 1)
    gaussian_kernel_2D = signal.convolve2d(vertical_kernel, horizontal_kernel)

    # Blur image
    smoothed_img = cv2.filter2D(image, cv2.CV_8U, gaussian_kernel_2D)

    # Normalize to [0 254] values
    smoothed_norm = np.array(image.shape)
    smoothed_norm = cv2.normalize(smoothed_img, None, 0, 254, cv2.NORM_MINMAX)  # Leave the second argument as None

    return smoothed_norm


def load_data(route):
    with open(route, 'r', encoding='utf-8') as content:
        dataBase = json.load(content)
        dataList = dataBase["monsters"]
        resList = []

        # Save on resultant List only large monsters of our game (MHRise)
        for data in dataList:
            for game in data["games"]:
                if game["game"] == "Monster Hunter Rise" and data["isLarge"]:
                    resList.append(data)

    return resList


# Returns true if the file with file_name is in the project folder
def file_exists(file_name):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_directory, file_name)

    return os.path.isfile(file_path)


class ObjectDetectorApp:

    def __init__(self, root):

        # Create tk root
        self.root = root
        self.root.title("MHRise Object Detector")
        self.root.geometry("1450x700")

        # -------- START OF MAIN FRAME --------
        self.main_frame = tk.Frame(root)

        # Initial label and buttons
        label = tk.Label(self.main_frame, text="¡Welcome to Monster Hunter Rise Object Detector!")
        label.grid(row=0, columnspan=2, ipady=20)

        self.bStart = tk.Button(self.main_frame, text="Detect Objects", command=self.show_video)
        self.bStart.grid(row=1, column=0)

        stop = tk.Button(self.main_frame, text="Stop video", command=self.stop_video)
        stop.grid(row=1, column=2, padx=10)

        # Entry for the video name
        self.entry = tk.Entry(self.main_frame, width=30)
        self.entry.insert(tk.END, "Type full video name (.mp4)")
        self.entry.grid(row=1, column=1)

        # Size for canvas
        self.canvas_width = 960
        self.canvas_height = 540

        # Create a canvas for the detections video
        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_width, height=self.canvas_height)
        self.canvas.grid(row=2, column=0)

        # Post-Detection label and buttons
        self.label1 = tk.Label(self.main_frame, text="Check Detections:")
        self.bIcon1 = tk.Button(self.main_frame, text="Monster 1", command=lambda: self.monsterFrame(0))
        self.bIcon2 = tk.Button(self.main_frame, text="Monster 2", command=lambda: self.monsterFrame(1))
        self.bIcon3 = tk.Button(self.main_frame, text="Monster 3", command=lambda: self.monsterFrame(2))
        self.bMap = tk.Button(self.main_frame, text="Map", command=self.frameMapa)

        # Show Main Frame
        self.main_frame.grid(row=0, column=0)

        # -------- END OF MAIN FRAME --------

        # -------- START OF SECONDARY FRAME --------
        # Initialize secondary frame
        self.secondary_frame = tk.Frame(root)
        # Create Canvas widget in the secondary frame
        self.image_canvas = tk.Canvas(self.secondary_frame, width=self.canvas_width, height=self.canvas_height)
        self.image_canvas.grid(row=2, column=0)

        # Back Button
        bBack = tk.Button(self.secondary_frame, text="Back", command=self.go_to_Main)
        bBack.grid(row=1, columnspan=2)

        self.text_area = tk.Text(self.secondary_frame, height=10, width=100)
        self.text_area.grid(row=4, column=0, padx=10, pady=10)

        # -------- END OF SECONDARY FRAME --------

        # Objects to process
        self.Icons = []
        self.Map = None

        # Won´t stop video until stop button is pressed.
        self.stop = False

        # Initialize canvas image references to future avoid of garbage collection
        self.image_reference_M = None
        self.photo_reference_M = None
        self.image_reference_S = None
        self.photo_reference_S = None

        # Initialize inference video thread
        self.video_thread = None

    def showMonster(self, i):
        # Given the icon index we search which monster match with this icon
        closerData, closerInGame = self.searchMonsterCoincidences(i)

        # Read the match image to show it on canvas
        image = cv2.imread(icons_path + closerInGame["image"])
        w, h = image.shape[1], image.shape[0]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(image)

        # Update the image on the existing canvas in the secondary frame
        self.image_canvas.config(width=w, height=h)
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        # Keep references to avoid garbage collection
        self.image_reference_S = image
        self.photo_reference_S = photo

        # Once we get the match we show information about the matched monster
        s = "In the map there is a {} called {}\n".format(closerData["type"], closerData["name"])
        s += closerInGame["info"]
        s += "\n"
        if "elements" in closerData:
            s += "It attacks with {} element\n".format(closerData["elements"])
        if "ailments" in closerData:
            s += "It may cause {} ailments\n".format(closerData["ailments"])
        s += "And Its weakness is {} elements\n".format(closerData["weakness"])
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, s)

    def showMap(self):
        # Having the map we search which map match with it
        closerData = self.searchMapCoincidences()

        # Read the match image to show it on canvas
        image = cv2.imread(maps_path + closerData["id"] + ".png")
        w, h = image.shape[1], image.shape[0]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(image)

        # Update the image on the existing canvas in the secondary frame
        self.image_canvas.config(width=w, height=h)
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        # Keep references to avoid garbage collection
        self.image_reference_S = image
        self.photo_reference_S = photo

        # Once we recognize the map, we show its information
        s = "We are in {}:\n".format(closerData["name"])
        s += closerData["text"]
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, s)

    def monsterFrame(self, i):
        if len(self.Icons) == 3:
            # Hide main frame and show secondary frame with monster of index i icon
            self.main_frame.grid_forget()
            self.secondary_frame.grid()
            self.showMonster(i)
        else:
            self.entry.delete(0, tk.END)
            self.entry.insert(tk.END, "Icons not found yet")

    def frameMapa(self):
        if self.Map is not None:
            # Hide main frame and show secondary frame with map
            self.main_frame.grid_forget()
            self.secondary_frame.grid()
            self.showMap()
        else:
            self.entry.delete(0, tk.END)
            self.entry.insert(tk.END, "Map not found yet")

    def go_to_Main(self):
        # Hide secondary frame and go back to main frame
        self.secondary_frame.grid_forget()
        self.main_frame.grid()

    def show_video(self):
        # We try showing the inference of our tflite model on a video
        try:
            # We get the name of the video of the entry
            s = self.entry.get()
            s += ".mp4"

            if file_exists(s):

                # We are starting an inference video, so stop has to be False
                self.stop = False

                # Hide post-detection frame Items
                self.label1.grid_forget()
                self.bIcon1.grid_forget()
                self.bIcon2.grid_forget()
                self.bIcon3.grid_forget()
                self.bMap.grid_forget()
                self.bStart.grid_forget()

                # Each time we detect we reset Icons and Map, for new detections.
                self.Icons = []
                self.Map = None

                # Start inference video thread
                self.video_thread = Thread(target=self.run_object_detection,
                                           args=(0.5,))
                self.video_thread.start()

            else:
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, "File not found")

        except Exception as e:
            print(f"Error: {e}")

    def stop_video(self):
        # Stop inference video
        self.stop = True

    def run_object_detection(self, min_conf_threshold):

        # Import TensorFlow libraries
        from tensorflow.lite.python.interpreter import Interpreter

        # WRITE HERE THE PATH OF YOUR WORKING DIRECTORY
        CWD_PATH = 'C:/tflite1'
        # ---------------------------------------------

        # s contains the path of the typed video, that should be on the working directory
        s = CWD_PATH
        s += '/'
        s += self.entry.get()
        s += '.mp4'

        # Path to video file
        VIDEO_PATH = s

        # WRITE HERE THE PATH TO .tflite FILE, WHICH CONTAINS THE MODEL THAT IS USED FOR OBJECT DETECTION
        PATH_TO_CKPT = 'C:/tflite1/TFLite_model/detect.tflite'

        # WRITE HERE THE PATH TO THE LABELMAP
        PATH_TO_LABELS = 'C:/tflite1/TFLite_model/labelmap.txt'

        # Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # Fix for label map if using the COCO "starter model" from
        # First label is '???', which has to be removed.
        if labels[0] == '???':
            del (labels[0])

        # Load the Tensorflow Lite model.
        interpreter = Interpreter(model_path=PATH_TO_CKPT)
        interpreter.allocate_tensors()

        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        floating_model = (input_details[0]['dtype'] == np.float32)

        input_mean = 127.5
        input_std = 127.5

        # Check output layer name to determine if this model was created with TF2 or TF1,
        # because outputs are ordered differently for TF2 and TF1 models
        outname = output_details[0]['name']

        if 'StatefulPartitionedCall' in outname:  # This is a TF2 model
            boxes_idx, classes_idx, scores_idx = 1, 3, 0
        else:  # This is a TF1 model
            boxes_idx, classes_idx, scores_idx = 0, 1, 2

        # Open video file
        video = cv2.VideoCapture(VIDEO_PATH)
        imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # find is True when we find the 3 Icons on the map
        found = False

        # We will show inference video until it ends, or we press the stop button
        while video.isOpened() and not self.stop:

            # Acquire frame and resize to expected shape [1xHxWx3]
            ret, frame = video.read()
            if not ret:
                print('Reached the end of the video!')
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[
                0]  # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index of detected objects
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence of detected objects

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)

                    # Draw label
                    object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window

                    # If the 3 icons aren´t found we will try to obtain them from this frame
                    if not found:
                        if object_name == 'icon' and len(self.Icons) != 3:
                            iconRegion = frame_rgb[ymin:ymax, xmin:xmax, :]
                            self.Icons.append(iconRegion)
                            if len(self.Icons) == 3:
                                found = True

                    # We search map with high confidence
                    if object_name == 'map' and scores[i] >= 0.99:
                        mapRegion = frame_rgb[ymin:ymax, xmin:xmax, :]
                        self.Map = mapRegion

                    cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                                  (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                                  cv2.FILLED)  # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                                2)  # Draw label text

            # If after processing this frame we haven´t found three icons, we will wait to other frame
            if len(self.Icons) < 3:
                self.Icons = []

            # All the results have been drawn on the frame, so it's time to display it in the canvas.
            # Resize each fotogram to canvas size
            frame_resized = cv2.resize(frame, (self.canvas_width, self.canvas_height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)

            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

            # Keep references to avoid garbage collection
            self.image_reference_M = image
            self.photo_reference_M = photo

        # Clean up
        video.release()

        # Now that the video has ended, we can show post-detection items
        self.label1.grid(row=3, column=1, padx=10)
        self.bIcon1.grid(row=3, column=2, padx=10)
        self.bIcon2.grid(row=3, column=3, padx=10)
        self.bIcon3.grid(row=3, column=4, padx=10)
        self.bMap.grid(row=3, column=5, padx=10)
        self.bStart.grid(row=1, column=0)

        # Reset entry to request new video name
        self.entry.delete(0, tk.END)
        self.entry.insert(tk.END, "Type full video name (.mp4)")

    def searchMonsterCoincidences(self, i):
        route = 'monsters.json'
        dataList = load_data(route)

        # Obtain a reference image
        for game in dataList[0]["games"]:
            # Nos quedamos con su imagen en la versión Rise
            if game["game"] == "Monster Hunter Rise":
                # Read image
                image = cv2.imread(icons_path + game["image"])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read selected Icon
        selected = self.Icons[i]

        # Gaussian filtering to reduce noise
        smoothed1 = gaussian_smoothing(selected, 2, 2)

        # Convert image to gray
        gray = cv2.cvtColor(smoothed1, cv2.COLOR_RGB2GRAY)

        # Let´s crop unnecessary margins:
        # First obtain the intensity of Icon background, by obtaining the pixel intensity with more frequency.
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        more_frequent_intensity = np.argmax(hist)

        # Now it´s crop time
        OFFSET = 60
        cropped = cropEdges(smoothed1, more_frequent_intensity, OFFSET)
        cropped = cropCorners(cropped, more_frequent_intensity, OFFSET)

        # Gaussian smoothing after crop
        cropped = gaussian_smoothing(cropped, 2, 2)

        # Resize to image reference for matching
        resized = cv2.resize(cropped, (image.shape[1], image.shape[0]))

        higherNCC = 0
        closerData = []
        closerInGame = []
        # Iterate all images and show the one which leads to higher ncc
        for i in range(len(dataList)):
            # Iterate all images in dataList
            for game in dataList[i]["games"]:
                # Compare with MHRise Icon image
                if game["game"] == "Monster Hunter Rise":
                    # Read image
                    image = cv2.imread(icons_path + game["image"])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Smoothe image before comparing
                    smoothedp = gaussian_smoothing(image, 2, 2)

                    ncc = cv2.matchTemplate(smoothedp, resized, cv2.TM_CCORR_NORMED)
                    if ncc[0][0] > higherNCC:
                        higherNCC = ncc
                        closerData = dataList[i]
                        closerInGame = game

        return closerData, closerInGame

    def countPixels(self, intensity):
        # Obtain image bounds
        height, width, _ = self.Map.shape

        # Reshape matrix to obtain list of pixels
        pixels = self.Map.reshape(height * width, 3)

        # Convert desired intensity in uint8
        intensity_uint8 = np.uint8(intensity)

        # OFFSET where pixels in the range are considered equals to the desired intensity
        OFFSET = 10

        # Count pixels which have their intensity on the desired range
        pixels_with_intensity = np.sum(
            np.all((pixels >= intensity_uint8 - OFFSET) & (pixels <= intensity_uint8 + OFFSET), axis=1))

        return pixels_with_intensity

    def searchMapCoincidences(self):
        route = 'maps.json'

        with open(route, 'r', encoding='utf-8') as content:
            dataBase = json.load(content)
            dataList = dataBase["maps"]

        proportionD = (self.Map.shape[1] / self.Map.shape[0])

        closerData = []
        # Iterate all images and show the one which leads to higher correspondence
        for data in dataList:
            # Read image
            image = cv2.imread(maps_path + data["id"] + ".png")

            # First we try to identify it with proportion differences
            proportionB = image.shape[1] / image.shape[0]
            difference = abs(proportionD - proportionB)

            if difference < 0.055:
                closerData.append(data)

        if len(closerData) == 1:
            return closerData[0]
        # If this isn´t enough to classify detected map, we try counting its sand, water and ground pixels
        else:
            # We have te RGB code of water,sand and ground pixels on the map
            waterPixel = [199, 226, 236]
            sandPixel = [171, 171, 137]
            groundPixel = [146, 143, 147]

            # Let´s count our detection pixels:
            nWater = self.countPixels(waterPixel)
            nSand = self.countPixels(sandPixel)
            nGround = self.countPixels(groundPixel)

            # The maps that can´t be classified by their bounds are sandy plains, frost island and lava caverns
            # Which have enough difference on their pixels of water sand and ground, so we can take advantage of this:

            # If there are more sand pixels, we are on sandy plains!
            if bigger(nSand, nWater, nGround):
                return dataList[1]
            # If there are more water pixels, we are on frost island!
            elif bigger(nWater, nSand, nGround):
                return dataList[3]
            # If there are more ground pixels, we are on lava cavern!
            else:
                return dataList[4]


# -------- MAIN --------
if __name__ == "__main__":
    # Create main window
    root = tk.Tk()
    app = ObjectDetectorApp(root)
    root.mainloop()
# -------- END OF MAIN --------
