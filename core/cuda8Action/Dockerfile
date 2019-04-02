FROM nvidia/cuda:8.0-runtime-ubuntu16.04

# Upgrade and install basic Python dependencies for this black-box action
RUN apt-get update && apt-get install -y bash \
    bzip2 \
    vim \
    curl \
    gcc \
    libc-dev \
    python-pip

RUN pip install --upgrade pip setuptools six
RUN pip install --no-cache-dir gevent==1.2.1 flask==0.12 requests==2.13.0

# Do not modify - this is the internal openwhisk invoker service port
ENV FLASK_PROXY_PORT 8080

RUN mkdir -p /actionProxy
ADD ../actionproxy.py /actionProxy/

RUN mkdir -p /action
ADD stub.sh /action/exec

RUN chmod +x /action/exec

CMD ["/bin/bash", "-c", "cd actionProxy && python -u actionproxy.py"]

