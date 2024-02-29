Sure! Here's the translation of the provided text into English:

---

We are going to use machine learning in Python and Tensorflow in this workshop to train an agent to play Mario Kart 64.

As a basis for this workshop, the project [Tensorkart](https://github.com/kevinhughes27/TensorKart) by **Kevin Hughes** was used. However, this project was already 8 years old and based on Python 2.7. The requirements needed for this project are not easily found in the standard package management sources. We have forked his project and made it suitable for Python 3.11.

You can find the fork of this repository here: https://github.com/WRKSHPZ/Tensorkart

## Prerequisites
We need [Python 3.11](https://www.python.org/downloads/release/python-3110/). This version is the sweet spot for most of the functionalities in the TensorKart project, except for **play.py**, which has a dependency on XFVB (virtual framebuffer) that I couldn't get to work on Windows 11.

We also need [Docker Desktop](https://www.docker.com/products/docker-desktop/) for the alternative we have for **play.py** to see your own model in action.

## Workshop intro
Since we are going to train a machine learning model to play Mario Kart 64, we will go through the 'standard' steps involved in training an ML model:
- Gathering and tagging training data
- Preparing training data for processing
- Training the model
- Verifying the performance of the model

This is not a simple model, but a [Tensorflow implementation](https://github.com/SullyChen/Autopilot-TensorFlow) of a model developed by Nvidia for controlling self-driving cars. You can read more about this model in this [whitepaper](https://arxiv.org/pdf/1604.07316.pdf).

## Step 0: Installing Prerequisites
You can find detailed information about installing the prerequisites here: [link](prerequisites.md)

TLDR:
- Install Python 3.11 (don't forget to adjust your PATH environment variable for both Python **and** Pip)
- Docker Desktop
- [RealVNC Viewer](https://www.realvnc.com/en/connect/download/viewer/)
- [Mupen64plus](https://github.com/mupen64plus/mupen64plus-core/releases/tag/2.5.9)
- [Mario Kart 64 ROM](https://mariokart.blob.core.windows.net/data/mario-kart-64.z64)
- Download repositories:
  - https://github.com/WRKSHPZ/Tensorkart
  - https://github.com/WRKSHPZ/gym-mupen64plus
- The script is 8 years old and therefore old-school for high-resolution screens, so it doesn't work well with resolutions above 1920x1080. If you have a higher resolution on your main screen, set it to a maximum of 1920x1080.

## Step 1: Gathering and Tagging Training Data
In the [Tensorkart](https://github.com/WRKSHPZ/Tensorkart) repository, you'll find several [python scripts](python-scripts.md). One of these scripts is **record.py**, which we'll use in the first step to generate our training data:

1. Connect a USB gamepad and check if it's recognized by the system.
2. Open the Tensorkart repository in Visual Studio Code.
3. Run a command to install Python dependencies: `pip install -r requirements.txt` (On Windows, you might run into an issue with path length: [link](https://docs.python.org/3.7/using/windows.html#removing-the-max-path-limitation))
4. Open a terminal and start record.py using: `python .\record.py`
5. A GUI will start. In the top left, you'll see 640x480 pixels from the top left corner of your screen. On the right, you'll see the input from the USB gamepad. At the bottom, there's a folder where the screenshots (frames) and joystick input will be saved, and a 'Record' button.
6. Now, copy the [Mario Kart 64 ROM](https://mariokart.blob.core.windows.net/data/mario-kart-64.z64) to the mupen64plus folder and start the mupen64plus emulator with the ROM in a new terminal in VS Code using: `.\mupen64plus-ui-console.exe mario-kart-64.z64`
7. Mario Kart 64 will now start in the emulator, and you'll see a spinning, golden Nintendo logo. You should position this emulator as closely as possible to the top left corner of your screen so that the GUI of **record.py** can collect as much data as possible.
8. Using the USB controller in Mario Kart 64, go to the Luigi Raceway time trials with Mario as the driver. To do this, press start until you see the 'Game select' screen. Start a 1p game, choose T.Trials, click begin, and click Ok. Select Mario as the driver, go to the Mushroom cup, and choose Luigi Raceway.
9. Once Lakitu (the turtle that signals the start) has disappeared, you can start recording. Click the 'Record' button in the **record.py** GUI and start racing.
10. You can complete the race entirely or stop earlier, but in machine learning, it's usually: the more data, the better. To stop, click the 'Stop' button in the **record.py** GUI.
11. The data is saved under the samples folder in the Tensorkart project under the folder you specified in the GUI. Check if the screenshots are there and open the **data.csv** file.

### Training Data
The **record.py** script has recorded the joystick's steering angle for each screen capture of the emulator in the **data.csv**.
In the data file, for each screenshot, you'll find a row with 6 columns of data, separated by commas:

```samples/2024-02-20_231645/img_0.png,0.00390625,0.00390625,0,0,0```

| Column | Example Data                      | Type                 | Value                                    |
| ------ | --------------------------------- | -------------------- | ---------------------------------------- |
| 1      | samples/2024-02-20_231645/img_0.png | String            | Screenshot filename                     |
| 2      | 0.00390625                         | Float between -1 and 1 | Normalized X-axis value of joystick     |
| 3      | 0.00390625                         | Float between -1 and 1 | Normalized Y-axis value of joystick     |
| 4      | 0                                  | 0 or 1               | A-button (acceleration)                 |
| 5      | 0                                  | 0 or 1               | RB-button (unused)                      |
| 6      | 0                                  | 0 or 1               | RT-button (unused)                      |

The Luigi Raceway circuit is 95% left turns. There is a script called **'flip-data.py'** among the python scripts. You can use this script

 to mirror screenshot and X-axis steering values, effectively doubling your training data!

### Training Tips
- Keep the acceleration pressed continuously for consistent training. The goal is to mainly train the X-axis value.
- Make smooth turns. You might tend to steer jerkily, but this means you have a sharp value (-1 for left or 1 for right) on 1 or 2 frames, with the intermediate frames registering no joystick value. The model will try to relate each frame to a steering value, so if you train more 0s than steering values, Mario will just go straight.
- Keep training data consistent. There are some visual settings you can adjust with the right analog stick. You can zoom out, show a speedometer, etc., but whatever you choose, ensure consistency in all your training data.
- Consider the effect of the GUI on training. Do you need a crop to exclude certain elements?
- Think about which circuits are convenient to train with? Should they be similar, or is it useful to train with very different circuits?
- Consider error situations. If Mario ever hits a wall, who will teach him to escape from that situation?
- To what extent are we training the model on the car's position? Might it be more accurate to remove the car and driver from the picture? If the model sees a screenshot of a car turning, it's given a certain steering angle. If we then train the model to turn when it sees the car turn, it will always go straight.
- Most Mario 64 tracks have more left turns than right turns. Should we take this into account?
- Should you stay in the middle of the track, or should you seek the edges? Should you train for both?
- Driving straight generates a lot of training data about driving straight. Does that perhaps influence the result too much? Should you consider removing the first and last training data frames from data.csv?
- Repeat the steps if you want to generate more training data, or supplement your training data with runs you can download from [this page](datasets.md)

## Step 2: Preparing Training Data for Processing
In the Tensorkart repository, you'll find several python scripts. One of these scripts is **utils.py**, which we'll use in this step to prepare our training data:

1. Open the Tensorkart repository in Visual Studio Code.
2. Open a terminal and start **utils.py** using: `python .\utils.py prepare samples/*`. With this command, all folders under samples are processed and added to the training data. If you want to specify a specific folder for training, use **samples/[folder name]**

Processing the training data creates two arrays of data using the NumPy library. One (**X.npy**) contains the screenshots converted into arrays of pixel values, and the other (**y.npy**) contains the joystick values we trained in step 1. By putting them into these arrays, Tensorflow can use them to train the model in step 3. An index on the array of screenshots in X corresponds to the index of the joystick values in Y.

## Step 3: Training the Model
Tensorflow is used to train our Mario model. We'll use the script **train.py** for this: `python .\train.py`
The script takes the npy files from the data folder and starts training.

When you open the **train.py** file, you'll find some very relevant values for our training job between lines 63 and 75:

**epochs**: The number of training cycles of your data. In each epoch, all training data is run through your model for training and validated with the percentage of the validation set (line 73: validation_split=0.2).

**batch_size**: The number of data points processed before your model is updated again. A large batch size can lead to overfitting. A small batch size results in a longer training time.

**learning_rate**: How fast your model 'learns' and is updated by your training data. It's a parameter that determines how quickly your model can change. A large value can lead to unstable training runs, and too small a value means not enough learning.

When training is complete (after all N epochs), you'll have an HDF5 model (**model_weights.h5**). This is a legacy model that is essentially deprecated by Tensorflow, but all scripts in this project work with it.

## Step 4: Verifying the Model's Performance
Don't get your hopes up just yet. If you've only made one set of training data and mirrored it, you probably don't have enough training data for a model to complete the circuit. Machine learning needs **a lot** of data, especially **a lot of high-quality** data.

You can download some [datasets](datasets.md) to add to your training data in subsequent runs. **Note: However, this data is not perfect either.**

Now, we want to see the model in action. Unfortunately, the **play.py** script is not in a working state.

For this, we need another repository: [gym-mupen64plus](https://github.com/WRKSHPZ/gym-mupen64plus). Gym stands for the OpenAI Gym wrapper, so it's a wrapper for the mupen64plus emulator that allows models to control agents in games. This is not limited to Mario Kart 64.

1. Go to the folder where you cloned the repo.
2. Copy the generated model (**model_weights.h5**) to the root. This will be used in the **example.py** to control the kart in the emulator in the Docker container.

The architecture looks like this (taken from the gym-mupen64plus documentation):
![Gym-Mupen64Plus architecture](image.png)
The red dashed lines are the container boundaries.

3. Now open a terminal and start the containers using: `docker-compose up --build -d` (note: this may take a few minutes the first time you build the containers).
4. The first build takes a bit longer due to fetching packages and dependencies, but once all 4 containers are started, open the RealVNC viewer and connect to `localhost:5900`
5. Via the VNC viewer, we can see what info is being sent about the Framebuffer container.

## Step 5: Lather, Rinse, Repeat
Now that we have the training pipeline set up, we can fine-tune the model so that it can complete a lap around the circuit. Check the [training tips](#training-tips) above for more guidance.

---

This should cover the translation of the entire text into English.
