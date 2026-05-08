import time
import serial
import serial.tools.list_ports
import threading
import json
from flask import Flask, jsonify, request, render_template, session, redirect, url_for, send_from_directory

# --- Configuration ---
# You can change this password
HARDCODED_PASSWORD = "admin" 

# --- App Setup ---
app = Flask(__name__)
# A Secret Key is required to use sessions
app.secret_key = 'your_very_secret_key_change_this_later' 

# --- Global State ---
# This dictionary will hold the latest data from the Arduino
latest_sensor_data = {
    "soil_moisture": 0,
    "pump_status": "OFF",
    "mode": "AUTO"
}
arduino_connected = False
arduino_serial = None

# --- Arduino Connection Logic (Runs in a Background Thread) ---

def find_arduino_port():
    """Finds the port connected to an Arduino."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "arduino" in port.description.lower() or "ch340" in port.description.lower():
            return port.device
    return None

def connect_to_arduino():
    """
    Manages the serial connection and reads data in a loop.
    This function runs in a separate thread.
    """
    global arduino_serial, arduino_connected, latest_sensor_data
    
    # We force COM6 based on our previous debugging
    ARDUINO_COM_PORT = "COM6" 
    BAUD_RATE = 9600

    while True:
        try:
            print(f"Attempting to connect to Arduino at {ARDUINO_COM_PORT}...")
            # Set a timeout so readline() doesn't block forever
            arduino_serial = serial.Serial(ARDUINO_COM_PORT, BAUD_RATE, timeout=3) # Increased timeout
            
            # Wait for the "ARDUINO_READY" signal or first line of JSON
            while True:
                # Use errors='ignore' to prevent crashes on bad bytes
                line = arduino_serial.readline().decode('utf-8', errors='ignore').strip()
                
                if not line:
                    # This means readline() timed out
                    print("Waiting for Arduino... (no signal received)")
                    continue

                print(f"Arduino says: {line}")
                
                # Check for "ARDUINO_READY" or a JSON string
                if line == "ARDUINO_READY" or (line.startswith('{') and line.endswith('}')):
                    print("Arduino connection established.")
                    arduino_connected = True
                    break # Exit the "waiting" loop
                
            # Now we are connected, start the main data-reading loop
            while True:
                line = arduino_serial.readline().decode('utf-8', errors='ignore').strip()
                
                if not line:
                    # Timeout, no data
                    continue
                
                if line.startswith('{') and line.endswith('}'):
                    try:
                        data = json.loads(line)
                        # Update the global state
                        latest_sensor_data = {
                            "soil_moisture": data.get("soil_moisture", 0),
                            "pump_status": data.get("pump_status", "OFF"),
                            "mode": data.get("mode", "AUTO")
                        }
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON: {line}")

        except serial.SerialException as e:
            print(f"Connection failed: {e}. Retrying in 5 seconds...")
            arduino_connected = False
            if arduino_serial:
                arduino_serial.close()
            time.sleep(5)
        except Exception as e:
            print(f"An error occurred: {e}. Retrying connection...")
            arduino_connected = False
            if arduino_serial:
                arduino_serial.close()
            time.sleep(5)

# --- Authentication Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles the login page."""
    error = None
    if request.method == 'POST':
        # Check if the password from the form is correct
        if request.form['password'] == HARDCODED_PASSWORD:
            # If correct, add 'logged_in' to the session
            session['logged_in'] = True
            # Send the user to the main dashboard
            return redirect(url_for('dashboard'))
        else:
            # If wrong, show an error
            error = 'Invalid password. Please try again.'
            
    # If a GET request, just show the login page
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    """Logs the user out."""
    # Remove 'logged_in' from the session
    session.pop('logged_in', None)
    # Send the user back to the login page
    return redirect(url_for('login'))

# --- Main Application Routes ---

@app.route('/')
def dashboard():
    """Serves the main dashboard page."""
    # Check if the user is logged in
    if 'logged_in' not in session:
        # If not, send them to the login page
        return redirect(url_for('login'))
        
    # If they are logged in, show the dashboard
    # Flask automatically looks in the 'templates' folder
    return render_template('index.html')

# --- Route for static files (style.css, script.js) ---
# This is crucial for the new folder structure
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


# --- API Routes (Protected) ---

def check_auth():
    """Helper function to protect API routes."""
    if 'logged_in' not in session:
        return (jsonify({"error": "Unauthorized"}), 401)
    return None

@app.route('/api/data', methods=['GET'])
def get_data():
    """API endpoint to get the latest sensor data."""
    auth_error = check_auth()
    if auth_error: return auth_error
    
    return jsonify({
        "connected": arduino_connected,
        "data": latest_sensor_data
    })

def send_arduino_command(command):
    """Helper function to send commands to the Arduino."""
    auth_error = check_auth()
    if auth_error: return auth_error
    
    if arduino_connected and arduino_serial:
        try:
            print(f"Sending command to Arduino: {command}")
            arduino_serial.write(f"{command}\n".encode('utf-8'))
            return jsonify({"status": "command_sent", "command": command})
        except Exception as e:
            print(f"Error writing to serial: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "error", "message": "Arduino not connected"}), 500

@app.route('/api/pump/on', methods=['POST'])
def pump_on():
    return send_arduino_command("MANUAL_ON")

@app.route('/api/pump/off', methods=['POST'])
def pump_off():
    return send_arduino_command("MANUAL_OFF")

@app.route('/api/pump/auto', methods=['POST'])
def pump_auto():
    return send_arduino_command("AUTO")

# --- Run the App ---
if __name__ == '__main__':
    # Start the background thread for the Arduino
    arduino_thread = threading.Thread(target=connect_to_arduino, daemon=True)
    arduino_thread.start()
    
    # Run the Flask web server
    app.run(host='0.0.0.0', port=5000)


