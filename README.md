# Parkinson-s-Disease-via-Deep-Learning-Model
This project develops an AI-based deep learning model to automatically detect Parkinson’s disease using facial images and videos. It analyzes reduced emotional facial expressions to classify PD and non-PD cases, providing a fast, accurate, and non-invasive solution for early diagnosis
Parkinson’s Disease Detection Using Deep Learning


An AI-based system for the automatic diagnosis of Parkinson’s disease using facial images and videos. The deep learning model analyzes reduced emotional facial expressions to classify Parkinson’s (PD) and non-Parkinson’s cases, offering a fast, accurate, and non-invasive solution for early detection.

📌 Project Overview

Parkinson’s disease is a neurological disorder that affects movement and facial expressions. A common symptom is facial masking (hypomimia), where facial muscles become stiff and emotional expressions are reduced.

This project uses deep learning and computer vision to automatically analyze facial expression patterns and predict Parkinson’s disease.

⚙️ System Workflow

Input Image/Video
→ Face Detection
→ Image Preprocessing
→ CNN Feature Extraction
→ PD / Non-PD Classification
→ Result Output

🚀 Key Features

✔ Automated facial emotion analysis
✔ Deep learning-based classification
✔ Non-invasive diagnosis
✔ Early detection support
✔ High accuracy prediction

🛠 Technologies Used

Python

OpenCV

TensorFlow / PyTorch

Convolutional Neural Networks (CNN)

NumPy, Matplotlib

📂 Dataset

Used facial expression datasets such as:

FER-2013

CK+ (Extended Cohn-Kanade)

Custom facial emotion images/videos

⚡ Installation
git clone https://github.com/yourusername/parkinsons-detection.git
cd parkinsons-detection
pip install -r requirements.txt

▶️ Run the System
python main.py

🧠 Train the Model
python train_model.py

📊 Model Performance
Metric	Value
Accuracy	94%
Precision	93%
Recall	92%
F1-Score	93%

(Values can be updated based on training results)

📸 Screenshots

Add your project screenshots here:

/screenshots
 ├── input_face.png
 ├── preprocessing.png
 ├── prediction_result.png


In README:

![Input](screenshots/input_face.png)
![Result](screenshots/prediction_result.png)

📈 Future Improvements

Real-time webcam detection

Mobile app deployment

Larger dataset training

Advanced CNN architectures

Clinical testing

Parkinsons-Detection-DeepLearning/
│
├── dataset/
│   ├── PD/
│   └── Non_PD/
│
├── models/
│   └── cnn_model.h5
│
├── src/
│   ├── face_detection.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict.py
│
├── main.py
├── requirements.txt
├── README.md
└── results/
    └── output_samples/

👨‍💻 Developed by

Samprith  N S

