FROM python:3.10-slim-bullseye

RUN apt-get update && apt-get upgrade -y

RUN pip3 --no-cache-dir install --upgrade awscli
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

COPY . /app

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]