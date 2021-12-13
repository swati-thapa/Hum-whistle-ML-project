FROM python:3.7.7

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

# execute the Flask app
CMD ["python", "app.py"]