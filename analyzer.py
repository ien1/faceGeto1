import cv2
from feat import Detector
import numpy as np
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from facelib import FaceDetector, AgeGenderEstimator

class FaceAnalyzer:
    def __init__(self, img) -> None:
        self.img = img
        self.h, self.w = self.img.shape[:2]

        # kernel for img sharpening
        self.kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])
        
        # Face detection and emotion analysis tool using py-feat
        self.detector = Detector(
            face_model="retinaface",
            landmark_model="mobilefacenet",
            au_model='svm',
            emotion_model="resmasknet",
            facepose_model="img2pose",
        )

        # MediaPipe for drawing facial landmarks
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # Age and gender estimation using facelib
        self.face_detector1 = FaceDetector()
        self.age_gender_detector1 = AgeGenderEstimator()


    def start_showing(self):
        # resizing image, sharpening and brightening
        width = self.w
        height = self.h
        if width > 600:
            width = 600
            height = (self.h / self.w) * width
        if height > 600:
            height = 600
            width = (self.w / self.h) * height
        self.img = cv2.resize(self.img, (int(width), int(height)))
        cv2.imshow("Face", self.img)
        #cv2.waitKey(2000)
        self.sharpen()
        self.increase_brightness()
        #cv2.imshow("Face", self.img)
        # write image to data/analyzed_face/face.jpg
        cv2.imwrite("data/analyzed_faces/face.jpg", self.img)
        # starting analysis
        self.detect_face()
        cv2.waitKey()
    
    def sharpen(self):
        # sharpening image
        self.img = cv2.filter2D(src=self.img, ddepth=-1, kernel=self.kernel)
    
    def increase_brightness(self, value=30):
        # increasing image brightness
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        self.img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    
    def detect_face(self):
        # face analysis

        # face detection and emotion analysis (without rendering anything on image)
        pred = self.detector.detect_image("data/analyzed_faces/face.jpg")

        # bounding box of detected image
        x1, y1, x2, y2 = int(pred["FaceRectX"][0]), int(pred["FaceRectY"][0]), int(pred["FaceRectX"][0]) + int(pred["FaceRectWidth"][0]), int(pred["FaceRectHeight"][0]) + int(pred["FaceRectX"][0])

        # emotions
        emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral", "Other"]
        vals = [0 for _ in emotions]
        i = 0
        while i < len(emotions) - 1:
            n = float(pred[emotions[i].lower()][0])
            # if value of actual emotion is below certain point, it is added to "other"
            if n < .01:
                vals[-1] += n
                del vals[i]
                del emotions[i]
            else:
                vals[i] += n
                i += 1


        # adding info border to right side of image
        h, w = self.img.shape[:2]
        vis = np.zeros((max(h, 600), w + 200,3), np.uint8)
        vis[:h, :w,:3] = self.img
        self.img = vis
        y1 = 30

        # age and gender estimation
        faces, boxes, scores, landmarks = self.face_detector1.detect_align(self.img)
        genders, ages = self.age_gender_detector1.detect(faces)
        
        # drawing results on info border
        self.img = cv2.putText(self.img, "Trait Analysis:", (w + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)  #COLOR: Title (facial analysis)
        self.img = cv2.rectangle(self.img, (w, 40), (w + 200, 65), (0, 150, 0), -1)  #COLOR: background for age and text (green)
        t = genders[0] + "".join(" " for _ in range(22 - len(genders[0]) - len(str(ages[0])))) + str(ages[0])
        self.img = cv2.putText(self.img, t, (w + 2, 57), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)  #COLOR:text (age and gender)

        # drawing results of emotion analysis
        y1 = 170
        self.img = cv2.putText(self.img, "Emotion Analysis:", (w + 2, 150), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)  #COLOR:Emotion analysis title
        for i in range(len(emotions)):
            # (x1, y1), (x2, y2)
            self.img = cv2.rectangle(self.img, (w, y1 + i * 25), (w + 200, y1 + i * 25 + 25), (20, 20, 20), -1)  #COLOR: background of bar
            self.img = cv2.rectangle(self.img, (w, y1 + i * 25), (w + int(vals[i] * 150) + 25, y1 + i * 25 + 25), (0, 150, 0), -1)  #COLOR:Foreground of bar
            self.img = cv2.rectangle(self.img, (w, y1 + i * 25), (w + 200, y1 + i * 25 + 25), (0, 0, 0), 1)  #COLOR:Border of bar
            self.img = cv2.putText(self.img, emotions[i] + "  " + str(vals[i] * 100)[:4] + "%", (w + 2, y1 + i * 25 + 17), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)  #COLOR:text on bar

        # drawing face landmarks
        with self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5) as face_mesh:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.img.flags.writeable = False
            results = face_mesh.process(self.img)
            self.img.flags.writeable = True
            self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=self.img,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()) #COLOR: find location of "get_default_face_mesh_tesselation_style" and change _GRAY to a tuple of desired colors

        # draw rect around face
        self.img = cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 150, 0), 2)  #COLOR: rect around image
        self.img = cv2.rectangle(self.img, (x1, y1), (x1 + 55, y1 + 22), (0, 150, 0), -1)  #COLOR: rect around text
        self.img = cv2.putText(self.img, str(pred["FaceScore"][0])[:5], (x1 + 3, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)  #COLOR: text



        # showing image
        cv2.imwrite("example images/analyze.png", self.img)
        cv2.imshow("Frame", self.img)




if __name__ == "__main__":
    img = cv2.imread("./data/imgs/annoyed.jpg", flags=cv2.IMREAD_COLOR)
    f = FaceAnalyzer(img)
    f.start_showing()