# Rumue AI - Thai Sign Language Recognition System

AI system สำหรับการจำแนกภาษามือภาษาไทย โดยใช้ Convolutional Neural Network (CNN) และ PyTorch

## 🌟 Features

- **Real-time Sign Language Recognition**: ระบบจำแนกภาษามือแบบผ่านกล้องเว็บแคม
- **GPU Acceleration**: รองรับการใช้งาน Intel Arc Graphics, CUDA, และ CPU
- **24 Letter Support**: รองรับตัวอักษร A-I และ K-Y (ยกเว้น J และ Z)
- **User-friendly Interface**: อินเทอร์เฟซง่ายต่อการใช้งาน

## 📋 Requirements

- Python 3.8+
- PyTorch 2.8.0+xpu (สำหรับ Intel Arc GPU)
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Pandas
- scikit-learn

## 🚀 Installation

1. Clone repository:
```bash
git clone https://github.com/yourusername/rumue_ai.git
cd rumue_ai
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/xpu
pip install opencv-python numpy matplotlib seaborn pandas scikit-learn kagglehub
```

## 📖 Usage

### Real-time Recognition

```bash
python webcam_test.py
```

### Training (Jupyter Notebook)

1. เปิด notebook:
```bash
jupyter lab "main.ipynb"
```

2. รันทุก cell เพื่อ train model ใหม่

## 🎯 Model Architecture

- **Input**: 28x28 grayscale images
- **Architecture**: 6-layer CNN with batch normalization
- **Output**: 24 classes (A-I, K-Y)



## 🎮 Controls

- **Q**: ออกจากโปรแกรม
- **Spacebar/S**: บันทึกภาพหน้าจอ
- วางมือในกรอบสี่เหลี่ยมสีน้ำเงินเพื่อให้ AI ทำนาย

## 📝 Supported Letters

✅ **รองรับ**: A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y

❌ **ไม่รองรับ**:
- **J**: ต้องใช้การเคลื่อนไหว (motion gesture)
- **Z**: ไม่มีใน Sign Language MNIST dataset

## 📁 Project Structure

```
rumue_ai/
├── README.md                              # Documentation
├── webcam_test.py                         # Real-time recognition script
├── model.pth                              # Trained model weights
├── mian.ipynb                             # Training notebook
└── LICENSE                                # License file
```

## 🔧 GPU Support

### Intel Arc Graphics
```python
device = torch.device('xpu:0')  # Auto-detected
```

### NVIDIA CUDA
```python
device = torch.device('cuda:0')  # Auto-detected
```

### CPU Fallback
```python
device = torch.device('cpu')    # Auto-fallback
```



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Sign Language MNIST Dataset by DataMunge
- PyTorch team for excellent deep learning framework
- Intel for XPU support



---
**Made with ❤️**
