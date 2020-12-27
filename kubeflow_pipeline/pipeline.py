import kfp
import kfp.components as comp
from kfp import dsl

@dsl.pipeline(
    name="mnist using arcface",
    description="CT pipeline"
)

def mnist_pipeline():
    data = dsl.ContainerOp(

    )
    train_model = dsl.ContainerOp(

    )
    embedding = dsl.ContainerOp(

    )
    train_faiss = dsl.ContainerOp(

    )
    validate = dsl.ContainerOp(

    )
    deploy = dsl.ContainerOp(

    )