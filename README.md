# Image-Processing-with-streamlit

## 🖼️ Streamlit Image Processing Studio

โปรเจ็คนี้เป็น Web App ที่สร้างด้วย Streamlit สำหรับการประมวลผลภาพแบบง่าย ๆ
สามารถเปิดภาพจาก Upload, Webcam, หรือ URL และทำ Image Processing พร้อมแสดงผลกราฟคุณสมบัติของภาพ

---

## ✨ ความสามารถ

### 📸 รองรับการโหลดภาพจาก:

- Upload (ไฟล์ .png, .jpg, .jpeg, .bmp, .webp)

- Webcam (ถ่ายรูปจากกล้องโน้ตบุ๊ค/เว็บแคมโดยตรง)

- URL (ลิงก์รูปจากอินเทอร์เน็ต)

### ⚙️ Image Processing ที่รองรับ:

- Grayscale

- Gaussian Blur

- Canny Edge Detection

- Brightness / Contrast

- Threshold (Binary)

- Sharpen

### 🎚️ สามารถปรับพารามิเตอร์ของแต่ละโหมดผ่าน GUI

- 👁️ แสดงภาพต้นฉบับ และภาพหลังการประมวลผลแบบ side-by-side

📊 แสดงกราฟ Histogram ได้ 2 แบบ:

- Gray Histogram

- RGB Histogram (แยกช่องสี R, G, B)

### 💾 ดาวน์โหลดภาพที่ประมวลผลแล้วเป็นไฟล์ .png

---

## 📦 การติดตั้ง

แนะนำให้ใช้ Python 3.9+

Clone หรือดาวน์โหลดโปรเจ็คนี้

ติดตั้ง dependencies:
```bash
pip install -r requirements.txt
```

🚀 การใช้งาน

รันแอปด้วยคำสั่ง:
```bash
streamlit run app.py
```
