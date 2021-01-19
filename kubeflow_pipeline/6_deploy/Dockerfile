FROM pytorch/torchserve:0.3.0-cpu
COPY . /app
WORKDIR /app
RUN pip install -r docker_requirements.txt
ENTRYPOINT ["python"]
CMD ["run.py"]