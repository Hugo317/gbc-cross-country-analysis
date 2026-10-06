FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "gunicorn dash_app:server -b 0.0.0.0:${PORT} --workers 2 --timeout 120"]
