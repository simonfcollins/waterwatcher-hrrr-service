FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends cron procps && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hrrr_controller.py /app/

CMD ["uvicorn", "hrrr_controller:app", "--host", "0.0.0.0", "--port", "8000"]
