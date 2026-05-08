# AI-Powered Smart Agriculture System 🌱

An integrated **AI + IoT based smart agriculture solution** that enables
early plant disease detection and automated irrigation to improve crop
health, reduce water usage, and support sustainable farming.

## 📌 Problem Statement
Traditional farming practices rely heavily on manual inspection and
fixed irrigation schedules, leading to:
- Late detection of plant diseases
- Over-irrigation or under-irrigation
- Excessive water wastage and crop loss

## 💡 Solution Overview
This project addresses these challenges using a **hybrid architecture**
that combines:

- **Computer Vision & Deep Learning**
  - CNN-based plant disease detection using **MobileNetV2**
  - Image-based classification of healthy vs diseased leaves

- **IoT-based Smart Irrigation**
  - Soil moisture-driven automated irrigation
  - Arduino-controlled relay and water pump

- **Web Dashboards**
  - Streamlit interface for disease prediction
  - Flask-based IoT dashboard for live monitoring and control

## 🧠 System Architecture

**Flow:**

Field Sensors → Arduino → AI Inference → Decision Logic → Pump Control  
                                     ↘ Web Dashboard

The system operates in real-time and supports both automatic and manual
irrigation control.

## 🛠️ Tech Stack

### AI & ML
- PyTorch/Tensorflow
- MobileNetV2 (transfer learning)
- OpenCV, Pillow
- Scikit-learn (evaluation)

### Web & Deployment
- Streamlit (AI interface)
- Flask (IoT dashboard)
- HTML, CSS, JavaScript

### IoT & Hardware
- Arduino Uno
- Soil Moisture Sensor (FC-28)
- Relay Module
- Submersible Water Pump


## 📊 Results & Performance

- **Plant disease detection accuracy:** ~94.4%
- **Water consumption reduction:** ~28%
- **Inference time:** Suitable for real-time usage
- **Irrigation logic:** Fully automated with manual override

Detailed graphs, metrics, and screenshots are available in the
[`results/`](results/) folder.

## 📁 Project Structure

ai-powered-smart-agriculture-system/
├── docs/ # Report, synopsis, presentation
├── software/ # AI model + Streamlit app
├── hardware/ # Arduino, sensors, wiring
├── iot_web_app/ # Flask backend + frontend
├── results/ # Metrics, graphs, screenshots
├── requirements.txt
└── README.md

## ▶️ How to Run (High Level)

1. Clone the repository
2. Install dependencies 

   pip install -r requirements.txt

3. Run AI interface

streamlit run software/plant_disease_detection/streamlit_app/app.py

4.Run IoT dashboard

python iot_web_app/backend/app.py
(Arduino must be connected and configured.)

🚧 Future Enhancements

Weather-based irrigation logic

Mobile application

Edge AI deployment (TensorRT / TFLite)

Multi-crop disease detection

👨‍🎓 Academic Context

This project was developed as a Major Project (Phase-II) under the
Department of Artificial Intelligence & Machine Learning,
SJB Institute of Technology, Bengaluru.

👤 Authors

Aasim Pasha
Benaka R N
Monish K
Sumukh S Murthy

📜 License

This project is licensed under the MIT License.