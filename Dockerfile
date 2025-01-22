FROM python:3.10

ARG HOST_UID=1000
ARG HOST_GID=1000

ENV LANG=C.UTF-8

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential

ENV APP_FOLDER=/app USER=app PYTHONPATH=$APP_FOLDER:$PYTHONPATH

RUN groupadd --force --gid $HOST_GID $USER && \
    useradd -r -m --uid $HOST_UID --gid $HOST_GID $USER

USER $USER

ENV PATH="/home/$USER/.local/bin:${PATH}"

WORKDIR $APP_FOLDER

# Install the app libraries
COPY --chown=$USER requirements.txt /tmp/requirements.txt
RUN pip install --upgrade --quiet pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm -rf /tmp/*

# Install the 'emmo' package
COPY --chown=$USER . .
RUN pip install . --verbose
