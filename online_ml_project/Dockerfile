FROM python:3.9-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY /data/model.joblib /data/model.joblib
COPY /data/heart_cleveland_upload.csv  /data/heart_cleveland_upload.csv
COPY app.py /app.py
COPY scr /scr
COPY app_params.py /app_params.py
COPY /test_app.py /test_app.py

WORKDIR .

ENV PATH_TO_MODEL="/model.joblib"

CMD ["uvicorn", "app:app_", "--host", "0.0.0.0", "--port", "8000"]