FROM python:3.11-slim

WORKDIR /service

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-4}"]
