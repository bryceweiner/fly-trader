FROM python:3.12-slim
WORKDIR /relay
COPY requirements.txt relay.py ./
RUN pip install --no-cache-dir -r requirements.txt
USER nobody
EXPOSE 8443
CMD ["python", "relay.py"]
