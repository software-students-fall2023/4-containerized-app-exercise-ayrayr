"""Machine Learning Client"""
# import packages for ml-client
import os
import time
import numpy as np
from dotenv import load_dotenv
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pymongo import MongoClient
# import certifi

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

load_dotenv()  # take environment variables from .env.

def initialize_database():
    """
    Initializes the database connection and returns the db object
    """
    cxn = MongoClient(os.getenv('MONGO_URI'), serverSelectionTimeoutMS=5000)
    try:
        db = cxn[os.getenv('MONGO_DBNAME')] # store a reference to the database
        print(' *', 'Connected to MongoDB!') # if we get here, the connection worked!
        return db
    except Exception as e:
        # the ping command failed, so the connection is not available.
        print(' *', "Failed to connect to MongoDB at", os.getenv('MONGO_URI'))
        print('Database connection error:', e) # debug
    return None

# MongoDB setup
# client = MongoClient(os.getenv("MONGO_URI"))
# db = client[os.getenv("MONGO_DBNAME")]
# collection = db[os.getenv("MONGO_COLLECTION")]

# client = MongoClient("localhost", 27017)
# db = client["emotion_db"]
# collection = db["records"]

# Create the emotion detection model
def get_model(weight_h5= "machine-learning-client/model.h5"):
    """
    Create emotion recognition model and load weight
    """
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
    model.load_weights(weight_h5)

    return model

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def detect_face(frame, model, db, emotion_dict={0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}):
    """
    The function that detect faces and emotions. Stores emotion into database.
    """
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

    if (db is not None):
        if faces.__len__()!=0:
            try:
                db.records.insert_one({"timestamp": time.ctime(), "emotion": emotion_dict[maxindex]})  
                print("emotion inserted to collection!")
            except Exception as e:
                print(f"Exception during insert_one: {e}")

    return frame
    
def run_ml():
    """
    Main function that runs the Emotion Detection System
    """
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # start the webcam feed
    cap = cv2.VideoCapture(0)

    db_connection = initialize_database()

    if db_connection is None:
        return
    
    model = get_model()

    while True:
        ret, frame = cap.read()

        this_frame = detect_face(
            frame, model, db_connection
        )

        cv2.imshow("Frame", this_frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    run_ml()