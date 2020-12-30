FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libgtk2.0-dev
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["run.py"]