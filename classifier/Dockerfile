FROM pytorch/pytorch

USER root

WORKDIR /wrkdir

COPY . .

RUN apt update -y \
    && apt upgrade -y \
    && apt install -y zip vim \
    && python -m pip install pipenv pipenv-shebang\
    && pipenv install
