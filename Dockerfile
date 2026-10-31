FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py WA_Fn-UseC_-HR-Employee-Attrition.csv ./

ENV PORT=8080
EXPOSE 8080

CMD ["python", "app.py"]
