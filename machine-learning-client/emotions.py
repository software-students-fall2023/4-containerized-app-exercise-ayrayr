# import packages for ml-client
import os
import time
import numpy as np
from dotenv import load_dotenv
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pymongo import MongoClient
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

load_dotenv()  # take environment variables from .env.

cxn = MongoClient(os.getenv('MONGO_URI'))
try:
    db = cxn[os.getenv('MONGO_DBNAME')] # store a reference to the database
    print(' *', 'Connected to MongoDB!') # if we get here, the connection worked!
    collection = db[os.getenv("MONGO_COLLECTION")]
except Exception as e:
    # the ping command failed, so the connection is not available.
    print(' *', "Failed to connect to MongoDB at", os.getenv('MONGO_URI'))
    print('Database connection error:', e) # debug


# MongoDB setup
# client = MongoClient(os.getenv("MONGO_URI"))
# db = client[os.getenv("MONGO_DBNAME")]
# collection = db[os.getenv("MONGO_COLLECTION")]

# client = MongoClient("localhost", 27017)
# db = client["emotion_db"]
# collection = db["records"]

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# load weight to the model
model.load_weights('machine-learning-client/model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# start the webcam feed
cap = cv2.VideoCapture(0)

# Variables for timing and result storage
saved_results = []

def detect_face():
    last_time_saved = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cur_time_saved = time.time()

        facecasc = cv2.CascadeClassifier('machine-learning-client/haarcascade_frontalface_alt.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # current_time = time.time()
        # if current_time - last_saved_time >= 10:
        #     if faces:
        #         last_emotion = emotion_dict[maxindex]
        #         saved_results.append(last_emotion)
        #         if len(saved_results) > 5:
        #             saved_results.pop(0)

        if (cur_time_saved-last_time_saved >= 10):
            if faces.__len__()!=0:
                saved_results.append(emotion_dict[maxindex])
                try:
                    collection.insert_one({"timestamp": time.ctime(), "emotion": emotion_dict[maxindex]})  
                except Exception as e:
                    print(f"Exception during insert_one: {e}")
                print("emotion inserted to collection!")
                last_time_saved = cur_time_saved

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(saved_results)

if __name__ == "__main__":
    detect_face()