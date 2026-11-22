FROM python:3.12-slim

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV CHECKPOINTS_DIR=/checkpoints DATA_DIR=/data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# /checkpoints 只读挂载旧检查点与新模型模板;/data 持久化已发布批次与注册表
