FROM python:3.8
WORKDIR /ml_app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6
COPY . .
EXPOSE 5001
CMD [ "python","emotions.py" ]