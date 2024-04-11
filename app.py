from flask import Flask, render_template, Response,jsonify,current_app,request,redirect,url_for
from imutils.video import VideoStream
from imutils.video import FPS
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image  
import pyautogui
import argparse
import mediapipe as mp
import time
import joblib
# from flask_pymongo import PyMongo
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, Text, JSON, ARRAY
from sqlalchemy.ext.declarative import declarative_base
import json
import threading
import time
from datetime import datetime
from sqlalchemy.sql import text

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import warnings
import PIL

# Suppress DeprecationWarnings raised by Pillow
warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
model_filename = 'posture_classifier_model.pkl'
loaded_model = joblib.load(model_filename)
posture_types = ['slouch', 'headforward', 'tilting', 'shoulders', 'leaning', 'normal']
last_posture_time = time.time()
predicted_posture="normal"
count=0

# app.config["MONGO_URI"] = "mongodb+srv://shehalshah264:nmmjkSsijDIwyb4i@cluster0.fpdocgu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# mongo = PyMongo(app)
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:rootpass@localhost/ipd"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

parser = argparse.ArgumentParser()

parser.add_argument('--input', default=0,
                    help='Path to the video to be processed or webcam id (0 for example). Default value: 0.')
parser.add_argument('--rectangle', default=False, action=argparse.BooleanOptionalAction,
					help='Show the rectangle around the detected face. Default: False')
parser.add_argument('--landmarks', default=False, action=argparse.BooleanOptionalAction,
					help='Show the landmarks of the detected face. Default: False')
parser.add_argument('--forehead', default=False, action=argparse.BooleanOptionalAction,
					help='Show the forehead in oridinal color. Default: False')
parser.add_argument('--forehead_outline', default=False, action=argparse.BooleanOptionalAction,
					help='Draw a rectangle around the detected forehead. Default: True')
parser.add_argument('--fps', default=False, action=argparse.BooleanOptionalAction,
					help='Show the framerate. Default: False')

dlib_keypoints_path     = "dependencies/shape_predictor_68_face_landmarks.dat"

font_path               = "dependencies/AvenirLTStd-Book.otf"
font_bold_path          = "dependencies/AvenirNextLTPro-Bold.otf"
font_size               = 24
font_bold_size          = 24
font                    = ImageFont.truetype(font_path, font_size)
font_bold               = ImageFont.truetype(font_bold_path, font_bold_size)
font_color              = (30, 30, 30)

text_upper_margin       = 70
text_left_margin        = 70
space_text_line_upper   = 60
space_text_line_lower   = 20
text_lines_separation   = 50
line_thickness          = 1
line_width              = 80
line_color              = (0, 0, 0)

keypoints_color         = (0, 255, 255)
bbx_color               = (0, 255, 0)

draw_bbx                = True
draw_landmarks          = True
draw_corners            = True
display_fps             = True

show_forehead           = True
forehead_offset         = 40
forehead_width          = 200
forehead_height         = 60
forehead_outline        = True
forehead_outline_thik   = 1
forehead_outline_color  = (255, 0, 0) 

uid=1

# class User:
#     def __init__(self, user_id):
#         self.user_id = user_id
#         self.posture = []
#         self.stress_level = []
#         self.eye_blinks = []

#     def add_posture(self, posture):
#         self.posture.append(posture)

#     def add_stress_level(self, stress_level):
#         self.stress_level.append(stress_level)

#     def add_eye_blinks(self, eye_blinks):
#         self.eye_blinks.append(eye_blinks)

#     def save(self):
#         user_data = {
#             "user_id": self.user_id,
#             "posture": self.posture,
#             "stress_level": self.stress_level,
#             "eye_blinks": self.eye_blinks
#         }
#         mongo.db.users.update_one({"user_id": self.user_id}, {"$set": user_data}, upsert=True)

# class User(db.Model):
#     user_id = db.Column(db.Integer, primary_key=True)
#     posture = db.Column(db.PickleType)
#     stress_level = db.Column(db.PickleType)
#     eye_blinks = db.Column(db.PickleType)

#     def __init__(self, user_id):
#         print("hiiii init")
#         self.user_id = user_id
#         self.posture = []
#         self.stress_level = []
#         self.eye_blinks = []

#     def add_posture(self, posture):
#         self.posture.append(posture)

#     def add_stress_level(self, stress_level):
#         self.stress_level.append(stress_level)

#     def add_eye_blinks(self, eye_blinks):
#         self.eye_blinks.append(eye_blinks)

#     def save(self):
#         with app.app_context():
#             print(self.posture)
#             db.session.add(self)
#             db.session.commit()

Base = declarative_base()

def get_current_session():
    engine = create_engine("mysql+pymysql://root:rootpass@localhost/ipd")
    Session = sessionmaker(bind=engine)
    return Session()

session = get_current_session()
class User(Base):
    __tablename__ = 'user'
    user_id = Column(Integer, primary_key=True)
    email = db.Column(db.String(255))
    password = db.Column(db.String(255))
    posture = Column(JSON)
    stress_level = Column(Text)
    eye_blinks = Column(ARRAY(Integer))

    def __init__(self, user_id):
        # session.add(self)
        self.user_id = user_id
        self.posture = {}
        self.stress_level = []
        self.eye_blinks = []

    def add_posture(self, posture):
        if posture in self.posture:
            self.posture[posture] += 1
        else:
            self.posture[posture] = 1

    def add_stress_level(self, stress_level):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        if(stress_level>10):
            self.stress_level.append({'timestamp': timestamp, 'stress_level': stress_level})
        print(self.stress_level,type(self.stress_level))

    def add_eye_blinks(self, eye_blinks):
        self.eye_blinks.append(eye_blinks)

    def calculate_average_blink_rate(self, period_minutes):
        if len(self.eye_blinks) < period_minutes:
            return None # Not enough data to calculate average
        return sum(self.eye_blinks[-period_minutes:]) / period_minutes

    def save(self):
        with app.app_context():
            # self.posture = json.dumps(self.posture)
            # self.stress_level = json.dumps(self.stress_level)
            # self.eye_blinks = json.dumps(self.eye_blinks)
            # Serialize the lists to JSON strings before saving
            # session.commit()
            # session.close()
            session = get_current_session()
            print(self.user_id)
            existing_user = session.query(User).filter_by(user_id=self.user_id).first()
            # print(existing_user)
            if existing_user:
                existing_user.eye_blinks = json.loads(existing_user.eye_blinks)
                existing_user.eye_blinks.extend(self.eye_blinks)
                existing_user.eye_blinks = json.dumps(existing_user.eye_blinks)
                existing_user.stress_level = json.loads(existing_user.stress_level)

                # print("beoferee",existing_user.posture, type(existing_user.posture))
                print("beoferee",existing_user.stress_level, type(existing_user.stress_level))
                # existing_user.posture = json.loads(existing_user.posture)
                # Merge the existing posture counts with the new ones
                # print("after",existing_user.posture, type(self.posture))
                existing_user.stress_level.extend(self.stress_level)
                print("mid",existing_user.stress_level, type(self.stress_level))
                existing_user.stress_level = json.dumps(existing_user.stress_level)
                print("after",existing_user.stress_level, type(self.stress_level))
                existing_user.posture = {**existing_user.posture, **self.posture}
                session.commit()
            else:
                session.add(self)
                session.commit()
            session.close()


# db.create_all()
with app.app_context():
    db.create_all()

class VideCapture:
    def __init__(self):
        self.vs         = None
        self.frame      = None
        self.gray       = None
        self.disp       = None
        self.disp_size  = None
        self.proc_size  = None
        self.flip       = True

    def is_open(self):
        return True

    def get_frame(self, flip=False, resize=None):
        self.frame = self.vs.read()
        if flip:
            self.frame = cv2.flip(self.frame, 1)
        if resize is not None and self.frame is not None and len(resize) == 2:
            self.frame = cv2.resize(self.frame, resize)

    def to_grayscale(self, image, resize=None):
        if image is None:
            print("Error: Input image is None")
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if resize is not None and len(resize) == 2:
            gray = cv2.resize(gray, resize)
        return gray

    def to_color(self, image):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def resize(self, frame, new_size):
        return cv2.resize(frame, new_size)

    def start(self, input_video, resize_input, resize_output):
        self.vs = VideoStream(src=input_video).start()
        self.proc_size = resize_input
        self.disp_size = resize_output

    def update(self):
        self.get_frame(flip=self.flip, resize=self.disp_size)
        self.gray  = self.to_grayscale(self.frame, resize=self.proc_size)
        self.disp  = self.to_color(self.to_grayscale(self.frame))


class Stress:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dlib_keypoints_path)
        self.forehead = None
        self.data_buffer = []
        self.times = []
        self.fft = []
        self.buffer_size = 250
        self.hri = 0
        self.wait = 0
        self.t0 = time.time()
        self.fps = 0
        self.stress = 0
        self.frame_resize = (320, 216)    # Resizing the input image to speed up the processing
        self.screen_width = pyautogui.size()[0] # Width of the screen
        self.screen_height = pyautogui.size()[1] # Height of the screen
        self.resize_factor_width = int(self.screen_width / self.frame_resize[0])
        self.resize_factor_height = int(self.screen_height / self.frame_resize[1])

        self.disp_forehead = show_forehead
        self.disp_forehead_outline = forehead_outline
        self.disp_fps = display_fps
        self.disp_landmarks = draw_landmarks
        self.disp_rectangle = draw_bbx


    def init(self):
        self.data_buffer = []
        self.times = []
        self.fft = []
        self.hri = 0
        self.wait = 0
        self.t0 = time.time()
        self.stress = 0

    def set_screen_size(self,width, Height):
        self.screen_width = width
        self.screen_height = Height

    def set_screen_size(self,new_size):
        self.set_screen_size(new_size[0], new_size[1])

    def get_faces(self, image):
        return self.detector(image, 0)

    def get_first_face(self, image):
        rect = None
        rects =self.get_faces(image)
        if len(rects) > 0:
            rect = rects[0]
        return rect

    def get_landmarks(self, image, rect):
        shape = self.predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
        return shape

    def get_forehead(self, image, landmarks):
        p8c = [landmarks[8][0], landmarks[8][1] - 2*(landmarks[8][1]-landmarks[29][1])]
        p27 = landmarks[27]

        forehead_p1 = (p27[0]*self.resize_factor_width-int(forehead_width/2), 
                       p8c[1]*self.resize_factor_height+forehead_offset)
        forehead_p2 = (p27[0]*self.resize_factor_width+int(forehead_width/2), 
                       p8c[1]*self.resize_factor_height+forehead_offset+forehead_height)
        forehead = image[forehead_p1[1]:forehead_p2[1], forehead_p1[0]:forehead_p2[0]]
        return forehead, forehead_p1, forehead_p2

    def display_forehead(self, display):
        self.disp_forehead = display

    def display_forehead_outline(self, display):
        self.disp_forehead_outline = display

    def display_fps(self, display):
        self.disp_fps = display

    def display_landmarks(self, display):
        self.disp_landmarks = display

    def display_rectangle(self, display):
        self.disp_rectangle = display


    def get_means(self, image):
        return (np.mean(image[:, :, 0]) + np.mean(image[:, :, 1]) + np.mean(image[:, :, 2])) / 3.

    def get_stress_info(self):
        self.times.append(time.time()-self.t0)
        self.vals = self.get_means(self.forehead)
        self.data_buffer.append(self.vals)
        L = len(self.data_buffer)
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size
        processed = np.array(self.data_buffer)
        if L > 10:
            # self.fps = float(L) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], L)
            interpolated = np.interp(even_times, self.times, processed)
            interpolated = np.hamming(L) * interpolated
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            fft = np.abs(raw)
            # print(L)
            # print(self.fps)
            freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * freqs
            freqs = freqs[1:]
            idx = np.where((freqs > 50) & (freqs < 180))
            pruned = fft[idx]
            phase = phase[idx]
            pfreq = freqs[idx]
            freqs = pfreq
            fft = pruned
            if pruned.any():
                idx2 = np.argmax(pruned)
                self.hri = freqs[idx2]
                self.wait = (self.buffer_size - L) / self.fps
            for i in range(0,10):
                self.stress = self.stress+self.hri
            self.stress = self.stress / 10.0

    def add_text_custom_font(self, image, text, position, font, color):
        # Pass the image to PIL
        pil_img = Image.fromarray(image)

        # Draw the text
        draw = ImageDraw.Draw(pil_img)
        draw.text(position, text, font=font, fill=color)

        return np.array(pil_img)

    def draw_rectangle(self, image, point1, point2, color, thikness, corners=60):
        cv2.rectangle(image, point1, point2, color, thikness)
        if corners is not None:
            (x1, y1) = point1
            (x2, y2) = point2
            cv2.line(image, (x1, y1), (x1+corners, y1), color, thikness+1)
            cv2.line(image, (x1, y1), (x1, y1+corners), color, thikness+1)

            cv2.line(image, (x1, y2-corners), (x1, y2), color, thikness+1)
            cv2.line(image, (x1, y2), (x1+corners, y2), color, thikness+1)

            cv2.line(image, (x2, y1), (x2-corners, y1), color, thikness+1)
            cv2.line(image, (x2, y1), (x2, y1+corners), color, thikness+1)

            cv2.line(image, (x2, y2), (x2-corners, y2), color, thikness+1)
            cv2.line(image, (x2, y2), (x2, y2-corners), color, thikness+1)

    def draw_landmarks(self, image, landmarslist):
        for (x, y) in landmarslist:
            cv2.circle(image, (x*self.resize_factor_width, y*self.resize_factor_height), 1, keypoints_color, -1)

    def run(self, input_video):
        global start_time
        cap = VideCapture()
        cap.start(input_video, self.frame_resize, (self.screen_width, self.screen_height))

        # cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)         
        # cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, 1)

        t0 = time.time()
        frame_num = 0
        fps_buffer = 10

        while True:
            cap.update()
            rect = self.get_first_face(cap.gray)
            if rect is not None:
                landmarks = self.get_landmarks(cap.gray, rect)

                self.forehead, forehead_p1, forehead_p2 = self.get_forehead(cap.frame, landmarks)
                if self.disp_forehead:
                    cap.disp[forehead_p1[1]:forehead_p2[1], forehead_p1[0]:forehead_p2[0]] = self.forehead
                if self.disp_forehead_outline:
                    cv2.rectangle(cap.disp, forehead_p1, forehead_p2, forehead_outline_color, forehead_outline_thik)

                # Drawing the rectangle on the face
                (x1, y1, x2, y2) = (rect.left()*self.resize_factor_width, 
                                    rect.top()*self.resize_factor_height, 
                                    rect.right()*self.resize_factor_width,
                                    rect.bottom()*self.resize_factor_height)
                if self.disp_rectangle:
                    self.draw_rectangle(cap.disp, (x1, y1), (x2, y2), color=bbx_color, thikness=1, corners=60)

                if self.disp_landmarks:
                    self.draw_landmarks(cap.disp, landmarks)

                if self.forehead is not None:
                    self.get_stress_info()

                    tmp_txt = "Heart rate imaging: "
                    tmp_txt_size = font.getsize(tmp_txt)[0]
                    cap.disp = self.add_text_custom_font(cap.disp, tmp_txt, 
                                                     position=(text_left_margin, text_upper_margin), font=font, color=font_color)
                    cap.disp = self.add_text_custom_font(cap.disp, "{:.2f}".format(self.hri), 
                                                     position=(text_left_margin+tmp_txt_size, text_upper_margin), 
                                                     font=font_bold, color=font_color)
                    tmp_txt_size += font_bold.getsize("{:.2f}".format(self.hri))[0]
                    tmp_txt = "bpm, wait "
                    cap.disp = self.add_text_custom_font(cap.disp, tmp_txt, 
                                                     position=(text_left_margin+tmp_txt_size, text_upper_margin), 
                                                     font=font, color=font_color)
                    tmp_txt_size += font.getsize(tmp_txt)[0]
                    cap.disp = self.add_text_custom_font(cap.disp, "{}".format(int(self.wait)), 
                                                     position=(text_left_margin+tmp_txt_size, text_upper_margin), 
                                                     font=font_bold, color=font_color)
                    tmp_txt_size += font_bold.getsize("{}".format(int(self.wait)))[0]
                    cap.disp = self.add_text_custom_font(cap.disp, "s", 
                                                     position=(text_left_margin+tmp_txt_size, text_upper_margin), 
                                                     font=font, color=font_color)

                    cap.disp = cv2.line(cap.disp, 
                                         (text_left_margin, text_upper_margin+space_text_line_upper), 
                                         (text_left_margin+line_width, text_upper_margin+space_text_line_upper), 
                                         color=line_color, thickness=line_thickness)

                    stress_txt = "Stress level: {:.2f}%".format(self.stress)
                    cap.disp = self.add_text_custom_font(cap.disp, stress_txt, 
                                                     position=(text_left_margin, text_upper_margin+space_text_line_upper+space_text_line_lower), 
                                                     font=font_bold, color=font_color)

                    current_time = time.time()
                    if current_time - start_time >= 60:
                        print("60 SECONDS HA")
                        # Assuming you have a user instance to update
                        global user
                        user.add_stress_level(self.stress)
                        user.save() # Ensure this method saves the user instance to the database
                        start_time = current_time

                    if self.disp_fps and self.fps is not None:
                        fps_txt = "{:.2f}fps".format(self.fps)
                        cap.disp = self.add_text_custom_font(cap.disp, fps_txt, position=(40, self.screen_height-40), font=font, color=font_color)

            else:
                self.init()

            # cv2.imshow('frame', cap.disp)
            ret, jpeg2 = cv2.imencode('.jpg', cap.disp)
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg2.tobytes() + b'\r\n')
            frame_num += 1
            if frame_num % fps_buffer == 0:
                self.fps = fps_buffer/(time.time()-t0)
                t0 = time.time()

            key = cv2.waitKey(1)
            if key == 113: # q
                break


EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 3

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

def extract_keypoints(image):
    if image is None:
        print("Error loading image")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        return [landmark.x for landmark in results.pose_landmarks.landmark] + [landmark.y for landmark in results.pose_landmarks.landmark]
    else:
        return None

def detect_posture():
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    global last_posture_time
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        current_time = time.time()
        if current_time - last_posture_time >= 2: 
            last_posture_time = current_time
            keypoints = extract_keypoints(frame)
            if keypoints:
                predicted_label = loaded_model.predict([keypoints])[0]
                global predicted_posture
                predicted_posture = posture_types[predicted_label]
                print(f"Predicted Posture: {predicted_posture}")
                cv2.putText(frame, f"Predicted Posture: {predicted_posture}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                # Assuming you have a user_id for the current user
                # user = User(1)
                # user=None
                # global count
                # if count == 0:
                #     count += 1 
                global user
                    
                user.add_posture(predicted_posture)
                user.save()
                ret, jpeg3 = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg3.tobytes() + b'\r\n')
            else:
                ret, jpeg3 = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg3.tobytes() + b'\r\n')

start_time = time.time()

def detect_blinks():
    global start_time
    COUNTER = 0
    TOTAL = 0
    print("Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("dependencies/shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    print("Starting video stream thread...")
    vs = VideoStream(src=0).start()
    fileStream = False
    time.sleep(1.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1

                COUNTER = 0

                    # Check if 30 seconds have passed
            current_time = time.time()
            if current_time - start_time >= 60:
                print("60 SECONDS HAVE PASSEDDDDDDD")
                # Assuming you have a user instance to update
                global user
                user.add_eye_blinks(TOTAL)
                user.save() # Ensure this method saves the user instance to the database
                TOTAL = 0
                start_time = current_time

            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Predicted Posture: {predicted_posture}", (10,150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

def update_posture_counts():
    while True:
        # Assuming `user` is a global variable holding the current user instance
        global user
        if user:
            user.save() # Save the updated posture counts to the database
        time.sleep(5) # Wait for 5 seconds before the next update

# Start the background thread to update posture counts every 5 seconds
threading.Thread(target=update_posture_counts, daemon=True).start()

args = parser.parse_args()
stress = Stress()
stress.display_rectangle(args.rectangle)
stress.display_landmarks(args.landmarks)
stress.display_forehead(args.forehead)
stress.display_forehead_outline(args.forehead_outline)
stress.display_fps(args.fps)

@app.route('/video_feed')
def video_feed():
    return Response(detect_blinks(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stress_feed')
def stress_feed():
    return Response(stress.run(args.input), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/posture_feed')
def posture_feed():
    return Response(detect_posture(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fetch_users')
def get_all_users():
    session = get_current_session()
    # users = session.query(User).all()
    users = session.query(User).filter_by(user_id=uid).first()

    
    # Convert User objects to dictionaries
    # users_dict = [{'posture': user.posture} for user in users]
    users_dict = {'posture': users.posture,'blinks':json.loads(users.eye_blinks),'stress':json.loads(users.stress_level)}

    return jsonify(users_dict)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.json
        email = data['email']
        password = data['password']

        posture_json = json.dumps({
            "normal": 0, 
            "slouch": 0, 
            "leaning": 0, 
            "tilting": 0, 
            "shoulders": 0, 
            "headforward": 0
        })

        # Insert new user into the database
        session = get_current_session()
        result=session.execute(
            text("""
            INSERT INTO user (email, password, posture, stress_level, eye_blinks)
            VALUES (:email, :password, :posture, :stress_level, :eye_blinks)
            """),
            {
                'email': email,
                'password': password,
                'posture': posture_json,
                'stress_level': '[]',
                'eye_blinks': '[]'
            }
        )
        session.commit() 

        global uid

        uid = result.lastrowid

        print(uid)

        with app.app_context():
            global user
            db.create_all() 
            user = User(uid)

        return jsonify({'status': 'success', 'redirect': url_for('dashboard')})

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        email = data['email']
        password = data['password']
        session = get_current_session()

        # Query the database for the user with the given email
        user1 = session.query(User).filter_by(email=email).first()

        with app.app_context():
            global user
            db.create_all() 
            user = User(user1.user_id)

        global uid

        uid = user1.user_id

        # Check if the user exists and the password matches directly
        if user1 and user1.password == password:
            # If the credentials are correct, return the user_id and redirect to the dashboard
            return jsonify({'status': 'success', 'user_id': user1.user_id, 'redirect': url_for('dashboard')})
        else:
            # If the credentials are incorrect, return an error message
            return jsonify({'status': 'error', 'message': 'Invalid email or password'})

    return render_template('login.html')

# def get_all_users():
#     session = get_current_session()
#     users = session.query(User).all()
#     # users = User.query.all()
#     print("use4rss",users)
#     return jsonify([{user} for user in users])

if __name__ == '__main__':
    with app.app_context():
        db.create_all() 
        user = User(uid)
    app.run(debug=True)
