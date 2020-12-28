import kfp
import kfp.components as comp
from kfp import dsl
from kubernetes import client as k8s_client

@dsl.pipeline(
    name="mnist using arcface",
    description="CT pipeline"
)

def mnist_pipeline():
    data_0 = dsl.ContainerOp(
        name="load & preprocess data pipeline",
        image="byeongjokim/mnist-pre-data:latest",
    )\
    .add_volume(k8s_client.V1Volume(name='data', host_path=k8s_client.V1HostPathVolumeSource(path='/data')))\
    .add_volume_mount(k8s_client.V1VolumeMount(mount_path='/data', name='data'))

    data_1 = dsl.ContainerOp(
        name="validate data pipeline",
        image="byeongjokim/mnist-val-data:latest",
    )\
    .add_volume(k8s_client.V1Volume(name='data', host_path=k8s_client.V1HostPathVolumeSource(path='/data')))\
    .add_volume_mount(k8s_client.V1VolumeMount(mount_path='/data', name='data'))

    data_1.after(data_0)

if __name__=="__main__":
    client = kfp.Client(host="http://220.116.228.93:8080/pipeline", namespace="kbj")
    client.create_run_from_pipeline_func(pipeline_func=mnist_pipeline, arguments={})   