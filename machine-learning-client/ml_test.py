import unittest
from unittest.mock import patch, MagicMock
import os
from emotions import app, model, MongoClient  

class TestMLClient(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # Test the Flask route
    def test_video_feed_route(self):
        response = self.app.get('/video_feed')
        self.assertEqual(response.status_code, 200)
        self.assertIn('multipart/x-mixed-replace', response.content_type)

    # Test MongoDB connection
    @patch('emotions.MongoClient')  
    def test_mongo_connection(self, mock_mongo_client):
        os.environ['MONGO_URI'] = 'mock_uri'
        os.environ['MONGO_DBNAME'] = 'mock_db'
        os.environ['MONGO_COLLECTION'] = 'mock_collection'

        mock_client = MagicMock()
        mock_db = mock_client['mock_db']
        mock_collection = mock_db['mock_collection']
        mock_mongo_client.return_value = mock_client

        collection = MongoClient(os.getenv('MONGO_URI')) 
        self.assertEqual(collection, mock_collection)

    # Test the machine learning model prediction
    @patch('cv2.VideoCapture')
    def test_model_prediction(self, mock_video_capture):
        mock_video_capture.return_value.read.return_value = (True, MagicMock())

        # Mocking the face detection and model prediction part
        with patch('cv2.CascadeClassifier') as mock_cascade_classifier:
            mock_cascade_classifier.return_value.detectMultiScale.return_value = [(10, 10, 100, 100)]

            with patch('cv2.resize') as mock_resize:
                mock_resize.return_value = MagicMock()

                with patch('numpy.expand_dims') as mock_expand_dims:
                    mock_expand_dims.return_value = MagicMock()

                    with patch('emotions.model.predict') as mock_predict:  
                        mock_predict.return_value = [[0, 0, 0, 1, 0, 0, 0]]

                        # Call your face detection function here
                        # Replace 'detect_face' with the actual function name
                        detect_face_result = detect_face()  
                        self.assertIsNotNone(detect_face_result)

if __name__ == '__main__':
    unittest.main()
