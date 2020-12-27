FROM python:3.6
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["run.py"]