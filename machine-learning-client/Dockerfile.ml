FROM python:3.8
WORKDIR /ml_app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD [ "python","emotions.py" ]