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

train_data_path = os.getenv('TRAIN_DATA', "/data/mnist/train")
test_data_path = os.getenv('TEST_DATA', "/data/mnist/test")
train_data_file = os.getenv("TRAIN_DATA_FILE", "train_mnist")
test_data_file = os.getenv("TEST_DATA_FILE", "test_mnist")

faiss_train_data_path = os.getenv('FAISS_TRAIN_DATA', "/data/faiss/train")
faiss_test_data_path = os.getenv('FAISS_TEST_DATA', "/data/faiss/test")
faiss_train_data_file = os.getenv('FAISS_TRAIN_DATA_FILE', "faiss_train")
faiss_test_data_file = os.getenv('FAISS_TEST_DATA_FILE', "faiss_test")

image_width = os.getenv("IMAGE_WIDTH", 28)
image_height = os.getenv("IMAGE_HEIGHT", 28)
image_channel = os.getenv("IMAGE_CAHNNEL", 1)
npy_interval = os.getenv("NPY_INTERVAL", 5000)