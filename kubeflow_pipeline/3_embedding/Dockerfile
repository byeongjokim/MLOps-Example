FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["run.py"]