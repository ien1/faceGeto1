from vidgear.gears import NetGear
import mediapipe as mp
import cv2
from deepface import DeepFace
import threading as t
import os
#from face import Face
import time

class App:
    def __init__(self, ip, port="5454"):

        # Mediapipe settings for face detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_detection = mp.solutions.face_detection

        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # Deepface parameters
        self.models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
        self.backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
        

        # Creating client to receive cam data from raspberry pi
        #self.client = NetGear(
        #    address=ip,
        #    port=port,
        #    protocol="tcp",
        #    pattern=1,
        #    receive_mode=True
        #)
        # capturing video
        self.cap = cv2.VideoCapture('./example images/sr2.mp4')

        self.run = False  # wether the app is updating the image
        self.main_frame = None  # Frame for the main screen

        self.detected_faces = []  # cv2 image objects
        self.face_rects = []  # coords
        self.results = []  # face recognition results
        self.thread = None

        self.fps = []
    
    
    def main_screen(self):
        # starting loop
        self.run = True
        frame = 0
        new = 1
        prev = 1
        with self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            while self.run:

                # receive frames from network
                #self.main_frame = self.client.recv()

                # read frame from vid
                ret, self.main_frame = self.cap.read()

                # check for received frame if Nonetype
                if self.main_frame is None:
                    break
                

                self.main_frame = cv2.resize(self.main_frame, (500, 250), interpolation=cv2.INTER_LINEAR)

                # face detection and recognition
                if frame == 0:
                    # detect faces
                    self.draw_rect(face_detection)
                    # recognize faces -> NOTE: see wich one is faster
                    self.rec_faces()
                    #self.thread = t.Thread(target=self.rec_faces)
                    #self.thread.start()

                # draw results
                self.draw_info()

                # update frame number
                frame += 1
                if frame == 8:
                    frame = 0

                # time for fps count
                new = time.time()
                fps = str(int(1 / (new - prev)))
                prev = new
                cv2.putText(self.main_frame, fps, (7, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (200,121,90), 3, cv2.LINE_AA)  #COLOR:


                # Display the image -> NOTE: comment this out if you want to just get the resulting image
                cv2.imshow("Output Frame", self.main_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.run = False

        #print(max(self.fps))
        cv2.destroyAllWindows()
        #self.client.close()  # comment out when not reading from vid
    
    
    def draw_rect(self, face_detection):
        # draw rects for detected faces
        self.main_frame = cv2.cvtColor(self.main_frame, cv2.COLOR_BGR2RGB)
        self.main_frame.flags.writeable = False
        results = face_detection.process(self.main_frame)
        self.main_frame.flags.writeable = True
        self.main_frame = cv2.cvtColor(self.main_frame, cv2.COLOR_RGB2BGR)
        self.detected_faces = []
        self.face_rects = []
        if results.detections:
            for detection in results.detections:
                loc_data = detection.location_data.relative_bounding_box
                h, w = self.main_frame.shape[:2]
                x1, y1, x2, y2 = int(loc_data.xmin * w), int(loc_data.ymin * h), int(w * (loc_data.xmin + loc_data.width)), int(h * (loc_data.ymin + loc_data.height))
                self.face_rects.append(((x1, y1), (x2, y2)))
                #self.main_frame = cv2.rectangle(self.main_frame, (x1, y1), (x2, y2),  	(200,121,90), 1)
                self.detected_faces.append(((x1, y1), self.main_frame[int(y1*0.7):int(y2*1.3), int(x1*0.7):int(x2*1.3)]))
    
    
    def draw_info(self):
        # draw name of face
        for p1, p2 in self.face_rects:
            cv2.rectangle(self.main_frame, p1, p2, (200,121,90), 1)  #COLOR:
        for coords, name in self.results:
            coords = (coords[0], coords[1] - 10)
            self.main_frame = cv2.rectangle(self.main_frame, (coords[0], coords[1] - 20), (coords[0] + 130, coords[1] + 10),  	(200,121,90), -1)  #COLOR:
            self.main_frame = cv2.putText(self.main_frame, name, (coords[0] + 3, coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (238,238,238), 1)  #COLOR:
    
    def rec_faces(self):
        # recognize faces:
        # starting thread for each face
        self.results = []
        threads = []
        for coords, i in self.detected_faces:
                threads.append(t.Thread(target=self.rf, args=(i,coords,)))
                threads[-1].start()
    
    def rf(self, i, coords):
        # recognize face in boundaries "coords"
        #try:
        df = DeepFace.find(img_path=i,
                                enforce_detection=False,
                                prog_bar=False,
                                db_path="./data/imgs/",
                                model_name="SFace")
        #except Exception as e:
        #    self.results.append((coords, "Unknown"))

        # good models: Facenet - some time to load, SFace, (ArcFace), 
        if "identity" in df.columns and df["identity"].any():
            df = df.sort_values(by=df.columns[1], ascending=False)
            identity = df["identity"][0]
            percentage = (1 - df[df.columns[1]][0]) * 100
            if percentage > 50:
                percentage = "  " + str(percentage)[:5] + "%"
                name = os.path.splitext(identity[identity.find("//")+2:])[0]
                self.results.append((coords, name[:-3] + percentage))
            else:
                self.results.append((coords, "Unknown"))
        else:
            self.results.append((coords, "Unknown"))
    


if __name__ == "__main__":
    ip = "192.168.178.22"  # ip of machine running this code
    app = App(ip=ip)
    app.main_screen()
