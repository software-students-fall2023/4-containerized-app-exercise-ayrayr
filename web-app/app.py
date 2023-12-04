"""Web App"""
from flask import Flask, render_template, request, redirect, url_for, make_response
from dotenv import load_dotenv
import os
import pymongo
from pymongo.mongo_client import MongoClient
import subprocess
from bson.objectid import ObjectId
import os

# instantiate the app
app = Flask(__name__)

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

db = initialize_database()

@app.route("/")
def index():
    """
    the main page of the web app
    """
    docs = db.records.find({}).sort("timestamp", -1).limit(5)
    print(docs)
    return render_template("index.html", docs=docs)

@app.route("/start_cam_feed")
def start_cam_feed():
    """
    start the live camera feed
    """
    try:
        subprocess.run(["python", "./machine-learning-client/emotions.py"])
        return redirect(url_for("index"))
    except Exception as e:
        return f"The camera cannot be opened: {str(e)}"

if __name__ == "__main__":
    PORT = os.getenv('PORT', 5002) # use the PORT environment variable, or default to 5000
    app.run(port=PORT, host="0.0.0.0")

