import kfp
import kfp.components as comp
from kfp import dsl

@dsl.pipeline(
    name="mnist using arcface",
    description="CT pipeline"
)

def mnist_pipeline():
    data_0 = dsl.ContainerOp(
        name="load & preprocess data pipeline",
        image="byeongjokim/mnist-pre-data:latest",
    )

    data_1 = dsl.ContainerOp(
        name="validate data pipeline",
        image="byeongjokim/mnist-val-data:latest",
    )

    data_1.after(data_0)

if __name__=="__main__":
    client = kfp.Client()
    client.create_run_from_pipeline_func(pipeline_func=soojin_pipeline, arguments={})