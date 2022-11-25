FROM python:3.8-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY data/model.joblib /model.joblib
COPY app.py /app.py
COPY scr /scr

WORKDIR .

ENV PATH_TO_MODEL="/model.joblib"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "15000"]