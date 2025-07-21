# Partial Face Recognition with Deep Learning

This final year undergraduate project explores facial recognition under real-world constraints by developing a deep learning system capable of identifying **partially visible faces**. Built on a fine-tuned **VGG16 architecture**, the system was trained on the LFW and PartialLFW datasets using **transfer learning**, **facial alignment**, and **data augmentation** to improve robustness against occlusions such as masks, sunglasses, and off-angle poses.

The project was completed as part of a BSc in Computing (Cybersecurity) at the University of Buckingham under the supervision of Dr. Naseer Al-Jawad.

---

## 📌 Key Highlights

- Developed a **custom face recognition pipeline** capable of handling **partial occlusions**
- Fine-tuned a pre-trained **VGG16 model** with TensorFlow/Keras
- Applied **transfer learning** and replaced fully connected layers for task-specific adaptation
- Integrated **Dlib facial landmark detection** and **OpenCV** preprocessing
- Trained on both **LFW** and **PartialLFW** datasets
- Achieved **50.16% accuracy** on occluded faces — with clear potential for further refinement
- Tackled technical challenges including hardware limitations, shape mismatches, and environment setup

---

## 🧠 Technical Stack

- **Python 3.8**
- **TensorFlow + Keras**
- **OpenCV + Dlib** for alignment and preprocessing
- **NumPy / Pandas / Scikit-learn** for data manipulation and evaluation

---

## 📊 Dataset Summary

- **Labelled Faces in the Wild (LFW)** – standard facial recognition dataset
- **PartialLFW** – enhanced dataset with occlusions (e.g. masks, glasses, angles)

---

## 🧪 Methodology Overview

1. **Face Alignment** using Dlib landmarks
2. **Preprocessing** to resize (224x224), normalize, and augment images
3. **Transfer Learning** from ImageNet-pretrained VGG16
4. **Model Fine-Tuning** and classification via custom fully connected layers
5. **Training with Early Stopping** and checkpointing
6. **Evaluation** using accuracy, F1 score, and validation graphs

---

## 📁 Repository Structure

Project-partial-face-recognition/
├── align/ # Alignment scripts and resources
├── input_folder/ # Sample images
├── partial_faces/ # Region-extracted data (eyes, nose, etc.)
├── scripts/
│ ├── align_dataset.py
│ ├── gen_partial_faces.py
│ ├── preprocess_images.ipynb
│ ├── split_dataset.ipynb
│ └── training_vgg16.ipynb
├── README.md
├── requirements.txt
└── final_report.pdf # Full academic write-up 


---

## 🚀 Getting Started

```bash
pip install -r requirements.txt
python scripts/align_dataset.py
python scripts/gen_partial_faces.py
jupyter notebook scripts/training_vgg16.ipynb

🎓 Academic Context
Project Title: Partial Face Recognition 
Author: Julia Szostakiewicz
Degree: BSc (Hons) Computing (Cybersecurity)
Award: First Class Honours
Institution: University of Buckingham
Supervisor: Dr Naseer Al-Jawad
