
import dlib
import time
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2
from imutils import face_utils

# Define missing variables
dlib_keypoints_path 	= "dependencies/shape_predictor_68_face_landmarks.dat"

class Stress:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dlib_keypoints_path)
        self.buffer_size = 250
        self.frame_resize = (320, 216)  # Resizing the input image to speed up the processing
        self.times = []
        self.t0 = time.time()
        self.forehead = np.zeros((self.frame_resize[1], self.frame_resize[0], 3), dtype=np.uint8)
        self.data_buffer = []
        self.fft = []
        self.hri = 0
        self.wait = 0
        self.fps = 0
        self.stress = 0
        self.frame_num = 0
        self.fps = 0
        self.fps_buffer = 10
        self.fps_last_time = time.time()
        
        
    def get_faces(self, image):
        return self.detector(image, 0)

    
    def process_frame(self, base64_str):
        print("hi")
        # Convert base64 string to image
        decoded_img = base64.b64decode(base64_str)
        img = Image.open(BytesIO(decoded_img))
        img_array = np.array(img)
        img_array = np.flip(img_array, axis=-1)  # Convert RGB to BGR format
        frame = cv2.resize(img_array, self.frame_resize)

        # Apply face detection and stress calculation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        landmarks_list = []
        self.frame_num += 1

        # Calculate fps
        if self.frame_num % self.fps_buffer == 0:
            current_time = time.time()
            elapsed_time = current_time - self.fps_last_time
            self.fps = self.fps_buffer / elapsed_time
            self.fps_last_time = current_time
        
        # self.frame_num += 1
        # if self.frame_num % fps_buffer == 0:
        #     self.fps = fps_buffer/(time.time()-t0)
        #     t0 = time.time()

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            landmarks_list.append(shape)

            # Get forehead region
            forehead, forehead_p1, forehead_p2 = self.get_forehead(frame, shape)
            self.forehead = forehead

        # Return processed frame and landmarks
        return frame, landmarks_list

    def get_means(self, image):
        if self.forehead is None:
            return 0  # Return some default value or handle the case appropriately
        else:
            return (np.mean(image[:, :, 0]) + np.mean(image[:, :, 1]) + np.mean(image[:, :, 2])) / 3.

    def get_stress_info(self, base64_str):
        # Convert base64 string to image
        decoded_img = base64.b64decode(base64_str)
        img = Image.open(BytesIO(decoded_img))
        img_array = np.array(img)
        img_array = np.flip(img_array, axis=-1)
        frame = cv2.resize(img_array, self.frame_resize)

        # Calculate stress-related information
        self.times.append(time.time() - self.t0)
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
            print(L)
            print(self.fps)
            freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * freqs
            freqs = freqs[1:]
            # print(freqs)
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
            for i in range(0, 10):
                self.stress = self.stress + self.hri
            self.stress = self.stress / 10.0
            print("Stress Level:", self.stress)
            return self.stress  # Return the calculated stress level
    def get_forehead(self, frame, landmarks):
        # Define the points for forehead region
        p8c = [landmarks[8][0], landmarks[8][1] - 2 * (landmarks[8][1] - landmarks[29][1])]
        p27 = landmarks[27]

        # Define the forehead region points
        forehead_p1 = (p27[0], p8c[1])
        forehead_p2 = (p27[0] + 200, p8c[1] + 60)

        # Extract forehead region from the frame
        forehead = frame[forehead_p1[1]:forehead_p2[1], forehead_p1[0]:forehead_p2[0]]

        return forehead, forehead_p1, forehead_p2

if __name__ == "__main__":
    stress_detector = Stress()
    base64_str = ""  # Placeholder, replace this with actual base64 string received from frontend
    frame, landmarks_list = stress_detector.process_frame(base64_str)
    stress_detector.get_stress_info()


# import dlib
# import time
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import base64
# import cv2
# from flask_socketio import SocketIO, emit
# from imutils import face_utils

# # Define missing variables
# dlib_keypoints_path = "../dependencies/shape_predictor_68_face_landmarks.dat"

# class Stress:
#     def __init__(self):
#         self.detector = dlib.get_frontal_face_detector()
#         self.predictor = dlib.shape_predictor(dlib_keypoints_path)
#         self.buffer_size = 250
#         self.frame_resize = (320, 216)  # Resizing the input image to speed up the processing
#         self.socketio = SocketIO()
#         self.times=[]
#         self.t0 = time.time()
#         self.forehead =np.zeros((self.frame_resize[1], self.frame_resize[0], 3), dtype=np.uint8)
#         self.data_buffer = []
#         self.fft = []
#         self.hri = 0
#         self.wait = 0
#         self.fps = 0
#         self.stress = 0
#         self.frame_resize = (320, 216)
        

# # 		self.screen_width = pyautogui.size()[0] # Width of the screen
# # 		self.screen_height = pyautogui.size()[1] # Height of the screen
# # 		self.resize_factor_width = int(self.screen_width / self.frame_resize[0])
# # 		self.resize_factor_height = int(self.screen_height / self.frame_resize[1])

# # 		self.disp_forehead = show_forehead
# # 		self.disp_forehead_outline = forehead_outline
# # 		self.disp_fps = display_fps
# # 		self.disp_landmarks = draw_landmarks
# # 		self.disp_rectangle = draw_bbx

#     def process_frame(self, base64_str):
#         # Convert base64 string to image
#         decoded_img = base64.b64decode(base64_str)
#         img = Image.open(BytesIO(decoded_img))
#         img_array = np.array(img)
#         img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert to BGR format
#         frame = cv2.resize(img_array, self.frame_resize)
        
#         # Apply face detection and stress calculation
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         rects = self.detector(gray, 0)
#         landmarks_list = []

#         for rect in rects:
#             shape = self.predictor(gray, rect)
#             shape = face_utils.shape_to_np(shape)
#             landmarks_list.append(shape)

#         # Return processed frame and landmarks
#         return frame, landmarks_list


#     def get_means(self, image):
#         print("hi")
#         if self.forehead is None:
#             print("knone")
#             return 0  # Return some default value or handle the case appropriately
#         else:
#             print("not none")
#             return (np.mean(image[:, :, 0]) + np.mean(image[:, :, 1]) + np.mean(image[:, :, 2])) / 3.

#     def get_stress_info(self, frame, landmarks):
#         self.times.append(time.time()-self.t0)
#         self.vals = self.get_means(self.forehead)
#         self.data_buffer.append(self.vals)
#         L = len(self.data_buffer)
#         if L > self.buffer_size:
#             self.data_buffer = self.data_buffer[-self.buffer_size:]
#             self.times = self.times[-self.buffer_size:]
#             L = self.buffer_size
#         processed = np.array(self.data_buffer)
#         if L > 10:
#             even_times = np.linspace(self.times[0], self.times[-1], L)
#             interpolated = np.interp(even_times, self.times, processed)
#             interpolated = np.hamming(L) * interpolated
#             interpolated = interpolated - np.mean(interpolated)
#             print(interpolated)
#             raw = np.fft.rfft(interpolated)
#             phase = np.angle(raw)
#             fft = np.abs(raw)
#             freqs = float(self.fps) / L * np.arange(L / 2 + 1)
#             freqs = 60. * freqs
#             freqs = freqs[1:]
#             idx = np.where((freqs > 50) & (freqs < 180))
#             pruned = fft[idx]
#             phase = phase[idx]
#             pfreq = freqs[idx]
#             freqs = pfreq
#             fft = pruned
#             if pruned.any():
#                 idx2 = np.argmax(pruned)
#                 self.hri = freqs[idx2]
#                 self.wait = (self.buffer_size - L) / self.fps
#             for i in range(0,10):
#                 self.stress = self.stress+self.hri
#             self.stress = self.stress / 10.0
#             print("Stress Level:", self.stress)

#         def run(self):
#             while True:
#             # Implement the main loop to continuously process frames
#             # For the sake of this example, let's assume we receive base64 encoded image string from somewhere
#                 base64_str = ""  # Placeholder, replace this with actual base64 string received from frontend
#                 processed_frame = self.process_frame(base64_str)
#                 cv2.imshow("Processed Frame", processed_frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

# # Entry point
# if __name__ == "__main__":
#     stress_detector = Stress()
#     stress_detector.run()





# /-----------------------------------------------------------------------------------------------------------------------------------------------
# /-----------------------------------------------------------------------------------------------------------------------------------------------
# /-----------------------------------------------------------------------------------------------------------------------------------------------
# /-----------------------------------------------------------------------------------------------------------------------------------------------
# /-----------------------------------------------------------------------------------------------------------------------------------------------
# /-----------------------------------------------------------------------------------------------------------------------------------------------
# /-----------------------------------------------------------------------------------------------------------------------------------------------
# /-----------------------------------------------------------------------------------------------------------------------------------------------
# /-----------------------------------------------------------------------------------------------------------------------------------------------
# /-----------------------------------------------------------------------------------------------------------------------------------------------
# /-----------------------------------------------------------------------------------------------------------------------------------------------
# /-----------------------------------------------------------------------------------------------------------------------------------------------
# /-----------------------------------------------------------------------------------------------------------------------------------------------
# /-----------------------------------------------------------------------------------------------------------------------------------------------


# import cv2
# import numpy as np
# import time
# from PIL import ImageFont, ImageDraw, Image  

# from imutils import face_utils
# import imutils
# import pyautogui

# import dlib


# dlib_keypoints_path 	= "dependencies/shape_predictor_68_face_landmarks.dat"

# ## Text
# # Fonts
# font_path               = "dependencies/AvenirLTStd-Book.otf"
# font_bold_path          = "dependencies/AvenirNextLTPro-Bold.otf"
# font_size 	 	 		= 24
# font_bold_size 	 		= 24
# font                    = ImageFont.truetype(font_path, font_size)
# font_bold               = ImageFont.truetype(font_bold_path, font_bold_size)
# font_color 	 	 		= (30, 30, 30)

# text_upper_margin 		= 70
# text_left_margin 		= 70
# space_text_line_upper   = 60
# space_text_line_lower   = 20
# text_lines_separation   = 50
# line_thickness 	 		= 1
# line_width 	 	 		= 80
# line_color 	 	 		= (0, 0, 0)


# # Face detection
# keypoints_color 	 	= (0, 255, 255)
# bbx_color 	 	 		= (0, 255, 0)

# draw_bbx 	 	 		= True
# draw_landmarks 	 		= True
# draw_corners 	 		= True
# display_fps             = True

# show_forehead           = True
# forehead_offset 	 	= 40
# forehead_width 	 		= 200
# forehead_height 	 	= 60
# forehead_outline 		= True
# forehead_outline_thik   = 1
# forehead_outline_color  = (255, 0, 0) # Blue



# class VideCapture:
# 	def __init__(self):
# 		self.cap        = None
# 		self.frame	    = None
# 		self.gray	    = None
# 		self.disp	    = None
# 		self.disp_size  = None
# 		self.proc_size  = None
# 		self.flip	    = True

# 	def is_open(self):
# 		return self.cap.isOpened()

# 	def get_frame(self, flip=False, resize=None):
# 		_, frame = self.cap.read()
# 		if flip:
# 			frame = cv2.flip(frame, 1)
# 		if resize is not None and len(resize) == 2:
# 			frame = cv2.resize(frame, resize)
# 		self.frame = frame

# 	def to_grayscale(self, image, resize=None):
# 		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 		if resize is not None and len(resize) == 2:
# 			gray = cv2.resize(gray, resize)
# 		return gray

# 	def to_color(self, image):
# 		return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# 	def resize(self, frame, new_size):
# 		return cv2.resize(frame, new_size)

# 	def start(self, input_video, resize_input, resize_output):
# 		self.cap = cv2.VideoCapture(input_video)
# 		self.proc_size = resize_input
# 		self.disp_size = resize_output

# 	def update(self):
# 		self.get_frame(flip=self.flip, resize=self.disp_size)
# 		self.gray  = self.to_grayscale(self.frame, resize=self.proc_size)
# 		self.disp  = self.to_color(self.to_grayscale(self.frame))


# class Stress:
# 	def __init__(self):
# 		self.detector = dlib.get_frontal_face_detector()
# 		self.predictor = dlib.shape_predictor(dlib_keypoints_path)
# 		self.forehead = None
# 		self.data_buffer = []
# 		self.times = []
# 		self.fft = []
# 		self.buffer_size = 250
# 		self.hri = 0
# 		self.wait = 0
# 		self.t0 = time.time()
# 		self.fps = 0
# 		self.stress = 0
# 		self.frame_resize = (320, 216)	 # Resizing the input image to speed up the processing
# 		self.screen_width = pyautogui.size()[0] # Width of the screen
# 		self.screen_height = pyautogui.size()[1] # Height of the screen
# 		self.resize_factor_width = int(self.screen_width / self.frame_resize[0])
# 		self.resize_factor_height = int(self.screen_height / self.frame_resize[1])

# 		self.disp_forehead = show_forehead
# 		self.disp_forehead_outline = forehead_outline
# 		self.disp_fps = display_fps
# 		self.disp_landmarks = draw_landmarks
# 		self.disp_rectangle = draw_bbx


# 	def init(self):
# 		self.data_buffer = []
# 		self.times = []
# 		self.fft = []
# 		self.hri = 0
# 		self.wait = 0
# 		self.t0 = time.time()
# 		self.stress = 0

# 	def set_screen_size(width, Height):
# 		self.screen_width = width
# 		self.screen_height = Height

# 	def set_screen_size(new_size):
# 		self.set_screen_size(new_size[0], new_size[1])

# 	def get_faces(self, image):
# 		return self.detector(image, 0)

# 	def get_first_face(self, image):
# 		rect = None
# 		rects =self.get_faces(image)
# 		if len(rects) > 0:
# 			rect = rects[0]
# 		return rect

# 	def get_landmarks(self, image, rect):
# 		shape = self.predictor(image, rect)
# 		shape = face_utils.shape_to_np(shape)
# 		return shape

# 	def get_forehead(self, image, landmarks):
# 		p8c = [landmarks[8][0], landmarks[8][1] - 2*(landmarks[8][1]-landmarks[29][1])]
# 		p27 = landmarks[27]

# 		forehead_p1 = (p27[0]*self.resize_factor_width-int(forehead_width/2), 
# 					   p8c[1]*self.resize_factor_height+forehead_offset)
# 		forehead_p2 = (p27[0]*self.resize_factor_width+int(forehead_width/2), 
# 					   p8c[1]*self.resize_factor_height+forehead_offset+forehead_height)
# 		forehead = image[forehead_p1[1]:forehead_p2[1], forehead_p1[0]:forehead_p2[0]]
# 		return forehead, forehead_p1, forehead_p2

# 	def display_forehead(self, display):
# 		self.disp_forehead = display

# 	def display_forehead_outline(self, display):
# 		self.disp_forehead_outline = display

# 	def display_fps(self, display):
# 		self.disp_fps = display

# 	def display_landmarks(self, display):
# 		self.disp_landmarks = display

# 	def display_rectangle(self, display):
# 		self.disp_rectangle = display


# 	def get_means(self, image):
# 		return (np.mean(image[:, :, 0]) + np.mean(image[:, :, 1]) + np.mean(image[:, :, 2])) / 3.

# 	def get_stress_info(self):
# 		self.times.append(time.time()-self.t0)
# 		self.vals = self.get_means(self.forehead)
# 		self.data_buffer.append(self.vals)
# 		L = len(self.data_buffer)
# 		if L > self.buffer_size:
# 			self.data_buffer = self.data_buffer[-self.buffer_size:]
# 			self.times = self.times[-self.buffer_size:]
# 			L = self.buffer_size
# 		processed = np.array(self.data_buffer)
# 		if L > 10:
# 			# self.fps = float(L) / (self.times[-1] - self.times[0])
# 			even_times = np.linspace(self.times[0], self.times[-1], L)
# 			interpolated = np.interp(even_times, self.times, processed)
# 			interpolated = np.hamming(L) * interpolated
# 			interpolated = interpolated - np.mean(interpolated)
# 			raw = np.fft.rfft(interpolated)
# 			phase = np.angle(raw)
# 			fft = np.abs(raw)
# 			freqs = float(self.fps) / L * np.arange(L / 2 + 1)
# 			freqs = 60. * freqs
# 			freqs = freqs[1:]
# 			idx = np.where((freqs > 50) & (freqs < 180))
# 			pruned = fft[idx]
# 			phase = phase[idx]
# 			pfreq = freqs[idx]
# 			freqs = pfreq
# 			fft = pruned
# 			if pruned.any():
# 				idx2 = np.argmax(pruned)
# 				self.hri = freqs[idx2]
# 				self.wait = (self.buffer_size - L) / self.fps
# 			for i in range(0,10):
# 				self.stress = self.stress+self.hri
# 			self.stress = self.stress / 10.0

# 	def add_text_custom_font(self, image, text, position, font, color):
# 		# Pass the image to PIL
# 		pil_img = Image.fromarray(image)

# 		# Draw the text
# 		draw = ImageDraw.Draw(pil_img)
# 		draw.text(position, text, font=font, fill=color)

# 		return np.array(pil_img)

# 	def draw_rectangle(self, image, point1, point2, color, thikness, corners=60):
# 		cv2.rectangle(image, point1, point2, color, thikness)
# 		if corners is not None:
# 			(x1, y1) = point1
# 			(x2, y2) = point2
# 			cv2.line(image, (x1, y1), (x1+corners, y1), color, thikness+1)
# 			cv2.line(image, (x1, y1), (x1, y1+corners), color, thikness+1)

# 			cv2.line(image, (x1, y2-corners), (x1, y2), color, thikness+1)
# 			cv2.line(image, (x1, y2), (x1+corners, y2), color, thikness+1)

# 			cv2.line(image, (x2, y1), (x2-corners, y1), color, thikness+1)
# 			cv2.line(image, (x2, y1), (x2, y1+corners), color, thikness+1)

# 			cv2.line(image, (x2, y2), (x2-corners, y2), color, thikness+1)
# 			cv2.line(image, (x2, y2), (x2, y2-corners), color, thikness+1)

# 	def draw_landmarks(self, image, landmarslist):
# 		for (x, y) in landmarslist:
# 			cv2.circle(image, (x*self.resize_factor_width, y*self.resize_factor_height), 1, keypoints_color, -1)

# 	def run(self, input_video):
# 		cap = VideCapture()
# 		cap.start(input_video, self.frame_resize, (self.screen_width, self.screen_height))

# 		cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)		  
# 		cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, 1)

# 		t0 = time.time()
# 		frame_num = 0
# 		fps_buffer = 10

# 		while cap.is_open():
# 			cap.update()
# 			rect = self.get_first_face(cap.gray)
# 			if rect is not None:
# 				landmarks = self.get_landmarks(cap.gray, rect)

# 				self.forehead, forehead_p1, forehead_p2 = self.get_forehead(cap.frame, landmarks)
# 				if self.disp_forehead:
# 					cap.disp[forehead_p1[1]:forehead_p2[1], forehead_p1[0]:forehead_p2[0]] = self.forehead
# 				if self.disp_forehead_outline:
# 					cv2.rectangle(cap.disp, forehead_p1, forehead_p2, forehead_outline_color, forehead_outline_thik)

# 				# Drawing the rectangle on the face
# 				(x1, y1, x2, y2) = (rect.left()*self.resize_factor_width, 
# 									rect.top()*self.resize_factor_height, 
# 									rect.right()*self.resize_factor_width,
# 									rect.bottom()*self.resize_factor_height)
# 				if self.disp_rectangle:
# 					self.draw_rectangle(cap.disp, (x1, y1), (x2, y2), color=bbx_color, thikness=1, corners=60)

# 				if self.disp_landmarks:
# 					self.draw_landmarks(cap.disp, landmarks)

# 				if self.forehead is not None:
# 					self.get_stress_info()

# 					tmp_txt = "Heart rate imaging: "
# 					tmp_txt_size = font.getsize(tmp_txt)[0]
# 					cap.disp = self.add_text_custom_font(cap.disp, tmp_txt, 
# 													 position=(text_left_margin, text_upper_margin), font=font, color=font_color)
# 					cap.disp = self.add_text_custom_font(cap.disp, "{:.2f}".format(self.hri), 
# 													 position=(text_left_margin+tmp_txt_size, text_upper_margin), 
# 													 font=font_bold, color=font_color)
# 					tmp_txt_size += font_bold.getsize("{:.2f}".format(self.hri))[0]
# 					tmp_txt = "bpm, wait "
# 					cap.disp = self.add_text_custom_font(cap.disp, tmp_txt, 
# 													 position=(text_left_margin+tmp_txt_size, text_upper_margin), 
# 													 font=font, color=font_color)
# 					tmp_txt_size += font.getsize(tmp_txt)[0]
# 					cap.disp = self.add_text_custom_font(cap.disp, "{}".format(int(self.wait)), 
# 													 position=(text_left_margin+tmp_txt_size, text_upper_margin), 
# 													 font=font_bold, color=font_color)
# 					tmp_txt_size += font_bold.getsize("{}".format(int(self.wait)))[0]
# 					cap.disp = self.add_text_custom_font(cap.disp, "s", 
# 													 position=(text_left_margin+tmp_txt_size, text_upper_margin), 
# 													 font=font, color=font_color)

# 					cap.disp = cv2.line(cap.disp, 
# 										 (text_left_margin, text_upper_margin+space_text_line_upper), 
# 										 (text_left_margin+line_width, text_upper_margin+space_text_line_upper), 
# 										 color=line_color, thickness=line_thickness)

# 					stress_txt = "Stress level: {:.2f}%".format(self.stress)
# 					cap.disp = self.add_text_custom_font(cap.disp, stress_txt, 
# 													 position=(text_left_margin, text_upper_margin+space_text_line_upper+space_text_line_lower), 
# 													 font=font_bold, color=font_color)

# 					if self.disp_fps and self.fps is not None:
# 						fps_txt = "{:.2f}fps".format(self.fps)
# 						cap.disp = self.add_text_custom_font(cap.disp, fps_txt, position=(40, self.screen_height-40), font=font, color=font_color)

# 			else:
# 				self.init()

# 			cv2.imshow('frame', cap.disp)
# 			frame_num += 1
# 			if frame_num % fps_buffer == 0:
# 				self.fps = fps_buffer/(time.time()-t0)
# 				t0 = time.time()

# 			key = cv2.waitKey(1)
# 			if key == 113: # q
# 				break

# if __name__ == "__main__":
# 	stress = Stress()
# 	stress.run(input_video="../../001.mp4")
 
 
































































































































































# {% comment %} <!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <meta name="viewport" content="width=device-width, initial-scale=1.0">
# <title>Stress Detection</title>
# <style>
#     #videoCanvas {
#         width: 100%;
#         height: auto;
#         border: 1px solid black;
#     }
# </style>
# </head>
# <body>
# <video id="videoCanvas" autoplay playsinline></video>
# <div id="stressLevel">Stress Level: Loading...</div> <!-- Initial message -->
# <div id="status">Status: Connecting...</div> <!-- Connection status message -->

# <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.3.2/socket.io.js"></script>
# <script>
#     const socket = io.connect('http://localhost:5000');

#     socket.on('connect', function() {
#         console.log('Connected to server');
#         document.getElementById('status').innerText = 'Status: Connected'; // Update status message
#     });

#     socket.on('disconnect', function() {
#         console.log('Disconnected from server');
#         document.getElementById('status').innerText = 'Status: Disconnected'; // Update status message
#     });

#     // Listen for stress_update event
#     socket.on('stress_update', function(data) {
#         console.log('Received stress update:', data); // Log received data
#         // Update stress level on the webpage
#         document.getElementById('stressLevel').innerText = 'Stress Level: ' + data.stress_level.toFixed(2) + '%';
#     });

#     socket.emit('start_stress_detection');

#     // Access webcam and render feed onto canvas
#     const videoCanvas = document.getElementById('videoCanvas');
#     const constraints = {
#         video: true
#     };

#     async function init() {
#         try {
#             const stream = await navigator.mediaDevices.getUserMedia(constraints);
#             videoCanvas.srcObject = stream;
#         } catch (err) {
#             console.error('Error accessing webcam: ', err);
#         }
#     }

#     // Call init() to start accessing webcam
#     init();
# </script>
# </body>
# </html> {% endcomment %}