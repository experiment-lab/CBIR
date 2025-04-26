# 🖼️ CBIR – Content-Based Image Retrieval

**CBIR (Content-Based Image Retrieval)** is a Python-based project that retrieves similar images from a large digital image database using computer vision techniques. It focuses on extracting low-level features such as **color**, **shape**, and **texture** to represent images, enabling effective comparison and retrieval.

This implementation is based on the **WANG image dataset**, consisting of 1,000 categorized images grouped into 10 semantic classes.

---

## 📌 Overview

CBIR systems aim to automate the process of image search by analyzing visual features directly from image data, rather than relying on metadata or annotations.

This project evaluates the discriminative power of various features by measuring their effectiveness in distinguishing images across classes.

GUI images:

![image](https://user-images.githubusercontent.com/59763282/130036695-1c3dbe00-49de-4ff8-9660-e7baff1a880f.png)

![image](https://user-images.githubusercontent.com/59763282/130036862-3f9fefb9-af4e-4187-af37-afb9cc4f5374.png)

![image](https://user-images.githubusercontent.com/59763282/130036881-99e86ab1-1c45-49e5-b657-152a97d8fd37.png)

---

## 🔍 Features Extracted

- 🎨 **Color Features**: Histogram-based color distribution analysis.
- 🌀 **Texture Features**: Texture pattern recognition using spatial frequency.
- 🧩 **Shape Features**: Contour and boundary-based shape descriptors.

Each image is represented using a combination of these features, enabling robust image comparison.

---

## 🖼️ Dataset

- **WANG Image Database**  
  - Total Images: **1,000**
  - Categories: **10** (each with 100 similar images)
  - Images are pre-labeled to facilitate performance testing across distinct classes.

---

## ⚙️ Technologies Used

- **Python 3.x**
- **OpenCV** – for image processing and feature extraction
- **NumPy / SciPy** – for numerical operations
- **Matplotlib** – for result visualization
- **skimage** – for texture and shape features

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/experiment-lab/CBIR.git
cd CBIR


