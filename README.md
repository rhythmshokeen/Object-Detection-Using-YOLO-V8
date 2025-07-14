# 🧠 Real-Time Object Detection Using YOLOv8

This project demonstrates **real-time object detection** using your webcam and the powerful **YOLOv8** deep learning model. With the help of **OpenCV**, **Ultralytics**, and **cvzone**, the model can detect and label over 80 object classes from the COCO dataset live on your video feed.


# 🚀 Features

- 🔍 Real-time object detection using webcam
- 🎯 Uses YOLOv8 pre-trained weights (`yolov8m.pt`)
- 🧾 Displays class names and confidence scores
- 🖼️ Stylish bounding boxes using `cvzone`
- 📈 Live FPS (Frames Per Second) display


# 🛠️ Technologies Used

- [Python](https://www.python.org/)
- [OpenCV (cv2)](https://opencv.org/)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [cvzone](https://github.com/cvzone/cvzone)


# 📦 Installation

First, clone the repository:

```
bash
git clone https://github.com/imrhythm/yolov8-object-detection.git
cd yolov8-object-detection

Install the required Python libraries:
pip install ultralytics opencv-python cvzone

Simply run the following command:
python yolov8_webcam.py

Make sure your webcam is connected and accessible.
```


# 🧠 Model Details
This project uses the YOLOv8 Medium model:

File: yolov8m.pt

Dataset: COCO (Common Objects in Context)

Classes: 80+

You can replace it with other variants like yolov8n.pt (nano) for faster speed on lower-end machines.


# 📋 Example Detected Classes
Some examples of what the model can detect:

~ Person

~ Bicycle

~ Dog

~ Laptop

~ Pen

~ Cup

~ Car
...and many more (from the COCO dataset)!


# 🧼 Cleanup
~ Press ESC key to stop the webcam and exit the program gracefully.


# 📌 To-Do / Ideas for Expansion
~ Save detection snapshots

~ Add sound alerts on detection

~ Stream detection via a web app

~ Filter detection to specific objects only


# 📜 License
~ This project is open source under the MIT License.


# 🙌 Acknowledgements
~ Ultralytics YOLOv8

~ cvzone by Murtaza Hassan

~ OpenCV Python



Created with ❤️ by RHYTHM SHOKEEN
