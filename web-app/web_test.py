import unittest
from flask import Flask
import os
from app import app, cxn  
from dotenv import load_dotenv
import pymongo

class FlaskAppTestCase(unittest.TestCase):

    def setUp(self):
        # Set up the test client
        self.app = app.test_client()
        self.app.testing = True

        # Load environment variables for testing
        load_dotenv()

        # Set up test database connection
        self.cxn = pymongo.MongoClient(os.getenv('MONGO_URI'))
        self.db = self.cxn[os.getenv('MONGO_DBNAME')]
        self.collection = self.db[os.getenv("MONGO_COLLECTION")]

    def test_index_route(self):
        # Test the main page
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_env_variables(self):
        # Test if environment variables are loaded
        self.assertIsNotNone(os.getenv('MONGO_URI'))
        self.assertIsNotNone(os.getenv('MONGO_DBNAME'))
        self.assertIsNotNone(os.getenv('MONGO_COLLECTION'))
        self.assertIsNotNone(os.getenv('PORT'))

    def test_mongo_connection(self):
        # Test MongoDB connection
        try:
            self.cxn.admin.command('ping')
        except Exception as e:
            self.fail(f"MongoDB connection failed: {e}")

    def tearDown(self):
        # Close the test database connection
        self.cxn.close()

if __name__ == '__main__':
    unittest.main()
