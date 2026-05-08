# Backend Service

This backend is implemented using **Flask** and is responsible for
communication between the Arduino and the web dashboard.

## Responsibilities
- Read sensor data from Arduino via serial port
- Maintain current system state
- Expose REST APIs for frontend consumption
- Send control commands to Arduino

## Key Files
- **app.py** – Flask application entry point
- **routes.py** – API route definitions
- **serial_reader.py** – Arduino serial communication logic
- **config.py** – Configuration settings (ports, baud rate)