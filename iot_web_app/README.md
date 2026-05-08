# IoT Web Application

This web application provides a real-time dashboard for the
**AI-Powered Smart Agriculture System**.

## Features
- Live soil moisture monitoring
- Water pump status (ON/OFF)
- Manual and automatic irrigation control
- Communication with Arduino via serial interface

## Architecture

- **backend/**  
  Flask-based backend that:
  - Reads data from Arduino
  - Exposes REST APIs
  - Controls irrigation logic

- **frontend/**  
  Web interface built using:
  - HTML
  - CSS
  - JavaScript

## Notes
- The backend communicates with Arduino using serial communication.
- The dashboard updates data in near real-time.