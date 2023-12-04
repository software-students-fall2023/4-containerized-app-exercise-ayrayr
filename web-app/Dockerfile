FROM python:3.8
WORKDIR /web_app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5002
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]