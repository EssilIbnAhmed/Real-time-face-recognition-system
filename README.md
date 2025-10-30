# Real-Time Face Recognition System

This project is a **real-time face recognition system** built with **OpenCV**, **Face Recognition**, and **DeepFace**.  
It detects and recognizes faces from a webcam feed using preloaded face datasets.

---

## Features

- Real-time face detection and recognition
- Two recognition methods:
  - `code.py` — uses the `face_recognition` library (fast and lightweight)
  - `face.py` — uses `DeepFace` with the **VGG-Face** deep learning model
- Supports multiple people and labeled datasets
- Displays bounding boxes, names, and similarity scores
- Customizable thresholds for accuracy and performance

---

## Project Structure

face_recognition_project/
│
├── code.py # Face recognition using face_recognition library
├── face.py # Face recognition using DeepFace (VGG-Face model)
├── import.py # Checks if dataset paths exist
├── dataset/ # Folder containing people's face images
└── README.md # Documentation

yaml
Copy code

---

## How It Works

1. **Prepare your dataset**
   - Create a folder called `dataset`
   - Inside it, create one subfolder per person (e.g., `Essil`, `Chada`, `Hajer`)
   - Add 2–5 clear photos per person

   Example:
dataset/
├── Essil/
│ ├── 1.jpg
│ ├── 2.jpg
├── Chada/
│ ├── 1.jpg
├── Hajer/
│ ├── 1.jpg

markdown
Copy code

2. **Run the scripts**
- `python code.py` → Fast recognition using `face_recognition`
- `python face.py` → Deep learning recognition using `DeepFace`
- `python import.py` → Verifies dataset paths

3. **Press `q`** to close the webcam window.

---

## Installation

Make sure you have **Python 3.8+**, then install the required libraries:

```bash
pip install opencv-python face-recognition deepface numpy scipy
```
Example Output

When you run the script, the webcam will display faces with rectangles and names above recognized individuals.
If a face isn’t in the dataset, it will appear as Unknown.


License
This project is open source and free to use for learning and research purposes.
