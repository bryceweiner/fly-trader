FROM python:3.12-slim
WORKDIR /relay
COPY requirements.txt relay.py ./
RUN pip install --no-cache-dir -r requirements.txt
USER nobody
# glibc's thread cache keeps freed events from merging: with it on, the heap grows ~24 MB/h once the buffer is full
ENV GLIBC_TUNABLES=glibc.malloc.tcache_count=0
EXPOSE 8443
CMD ["python", "relay.py"]
