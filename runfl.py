from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import json
import base64
import cv2

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

from stressdetection.StressDetection2 import Stress
stress_obj = Stress()


@socketio.on('frame_data')
def handle_frame_data(data):
    print("hi")
    base64_str = data['base64_str']
    # print(base64_str)
    # Process the frame to extract facial landmarks
    frame, landmarks_list = stress_obj.process_frame(base64_str)
    # print(landmarks_list)
    
    for landmarks in landmarks_list:
        
        # Calculate stress level using the processed frame and landmarks
        stress_level = stress_obj.get_stress_info(base64_str)
        # Emit the stress level back to the frontend
        emit('stress_update', {'stress_level': stress_level}, namespace='/')



def gen():
    while True:
        print("hi")
        _, data = cv2.imencode('.jpg', frame)  # Assuming 'frame' contains the video frame received from the frontend
        base64_str = base64.b64encode(data).decode('utf-8')
        stress_level = stress_obj.get_stress_info(base64_str,)
        socketio.emit("stress_update", {"stress_level": stress_level}, namespace="/")
        yield b'--frame\r\nContent-Type: application/json\r\n\r\n' + json.dumps({'stress_level': 0}).encode() + b'\r\n'

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@socketio.on("connect", namespace="/")
def connect():
    print("Client connected")

@socketio.on("disconnect", namespace="/")
def disconnect():
    print("Client disconnected")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=3000)
























































































































# # from flask import Flask, render_template, Response
# # from flask_socketio import SocketIO
# # import json

# # app = Flask(__name__)
# # socketio = SocketIO(app, cors_allowed_origins="*")

# # # Import Stress class from stressdetection.py
# # from StressDetection import Stress

# # # Initialize the Stress class
# # stress_obj = Stress()

# # def gen():
# #     while True:
# #         # Perform stress detection and get stress level
# #         stress_level = stress_obj.get_stress_level()
        
# #         # Emit stress update via SocketIO
# #         socketio.emit("stress_update", {"stress_level": stress_level}, namespace="/")
# #         yield (b'--frame\r\n'
# #                b'Content-Type: application/json\r\n\r\n' + json.dumps({"stress_level": stress_level}).encode() + b'\r\n')

# # @app.route("/")
# # def index():
# #     return render_template("index.html")

# # @app.route("/video_feed")
# # def video_feed():
# #     return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# # @socketio.on("connect", namespace="/")
# # def connect():
# #     print("Client connected")

# # @socketio.on("disconnect", namespace="/")
# # def disconnect():
# #     print("Client disconnected")

# # if __name__ == "__main__":
# #     socketio.run(app, host="0.0.0.0", port=5000)