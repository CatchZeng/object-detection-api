FROM python:3.8-slim-buster

WORKDIR /app

RUN pip3 install --no-cache-dir Flask flask-cors pillow tensorflow

COPY . .
RUN rm -rf .vscode README.md Dockerfile Makefile

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]