FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hrrr_controller.py /app/

CMD ["uvicorn", "hrrr_controller:app", "--host", "0.0.0.0", "--port", "8000"]
