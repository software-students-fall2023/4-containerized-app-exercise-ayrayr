from flask import Flask, render_template, request, redirect, url_for, make_response
from dotenv import load_dotenv
import os
import pymongo
from bson.objectid import ObjectId
import os

# instantiate the app
app = Flask(__name__)

load_dotenv()  # take environment variables from .env.

cxn = pymongo.MongoClient(os.getenv('MONGO_URI'))
try:
    db = cxn[os.getenv('MONGO_DBNAME')] # store a reference to the database
    print(' *', 'Connected to MongoDB!') # if we get here, the connection worked!
    collection = db[os.getenv("MONGO_COLLECTION")]
except Exception as e:
    # the ping command failed, so the connection is not available.
    print(' *', "Failed to connect to MongoDB at", os.getenv('MONGO_URI'))
    print('Database connection error:', e) # debug

@app.route("/")
def index():
    """
    the main page of the web app
    """
    return render_template("index.html")

if __name__ == "__main__":
    PORT = os.getenv('PORT', 5002) # use the PORT environment variable, or default to 5000
    app.run(port=PORT)

