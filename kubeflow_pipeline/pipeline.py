import os

import kfp
import kfp.components as comp
from kfp import dsl
from kubernetes import client as k8s_client

@dsl.pipeline(
    name="mnist using arcface",
    description="CT pipeline"
)
def mnist_pipeline():
    vop_data = dsl.VolumeOp(
        resource_name ="data-pvc"
        name="pvc_data",
        storage_class="",
        modes=dsl.VOLUME_MODE_RWM,
        size="5Gi",
        volume_name="data-pv"
    )

    vop_model = dsl.VolumeOp(
        resource_name ="train-model-pvc"
        name="pvc_model",
        storage_class="",
        modes=dsl.VOLUME_MODE_RWM,
        size="5Gi",
        volume_name="train-model-pv"
    )

    data_0 = dsl.ContainerOp(
        name="load & preprocess data pipeline",
        image="byeongjokim/mnist-pre-data:latest",
        pvolumes={"/data": vop_data.volume}
    ).set_display_name('collect & preprocess data')
    # .add_volume(k8s_client.V1Volume(name='data', host_path=k8s_client.V1HostPathVolumeSource(path='/data')))\
    # .add_volume_mount(k8s_client.V1VolumeMount(mount_path='/data', name='data'))\
    
    data_1 = dsl.ContainerOp(
        name="validate data pipeline",
        image="byeongjokim/mnist-val-data:latest",
        pvolumes={"/data": vop_data.volume}
    ).set_display_name('validate data').after(data_0)
    # .add_volume(k8s_client.V1Volume(name='data', host_path=k8s_client.V1HostPathVolumeSource(path='/data')))\
    # .add_volume_mount(k8s_client.V1VolumeMount(mount_path='/data', name='data'))\

    train_model = dsl.ContainerOp(
        name="train embedding model",
        image="byeongjokim/mnist-train-model:latest",
        pvolumes={"/data": vop_data.volume, "/model": vop_model.volume}
    ).set_display_name('train model').after(data_1)
    # .add_volume(k8s_client.V1Volume(name='data', host_path=k8s_client.V1HostPathVolumeSource(path='/data')))\
    # .add_volume_mount(k8s_client.V1VolumeMount(mount_path='/data', name='data'))\
    # .add_volume(k8s_client.V1Volume(name='model', host_path=k8s_client.V1HostPathVolumeSource(path='/model')))\
    # .add_volume_mount(k8s_client.V1VolumeMount(mount_path='/model', name='model'))\

    embedding = dsl.ContainerOp(
        name="embedding data using embedding model",
        image="byeongjokim/mnist-embedding:latest",
        pvolumes={"/data": vop_data.volume, "/model": vop_model.volume}
    ).set_display_name('embedding').after(train_model)
    # .add_volume(k8s_client.V1Volume(name='data', host_path=k8s_client.V1HostPathVolumeSource(path='/data')))\
    # .add_volume_mount(k8s_client.V1VolumeMount(mount_path='/data', name='data'))\
    # .add_volume(k8s_client.V1Volume(name='model', host_path=k8s_client.V1HostPathVolumeSource(path='/model')))\
    # .add_volume_mount(k8s_client.V1VolumeMount(mount_path='/model', name='model'))\

    train_faiss = dsl.ContainerOp(
        name="train faiss",
        image="byeongjokim/mnist-train-faiss:latest",
        pvolumes={"/data": vop_data.volume, "/model": vop_model.volume}
    ).set_display_name('train faiss').after(embedding)
    # .add_volume(k8s_client.V1Volume(name='data', host_path=k8s_client.V1HostPathVolumeSource(path='/data')))\
    # .add_volume_mount(k8s_client.V1VolumeMount(mount_path='/data', name='data'))\
    # .add_volume(k8s_client.V1Volume(name='model', host_path=k8s_client.V1HostPathVolumeSource(path='/model')))\
    # .add_volume_mount(k8s_client.V1VolumeMount(mount_path='/model', name='model'))\
    

    analysis = dsl.ContainerOp(
        name="analysis total",
        image="byeongjokim/mnist-analysis:latest",
        pvolumes={"/data": vop_data.volume, "/model": vop_model.volume},
        file_outputs={
            "confusion_matrix": "/model/confusion_matrix.csv",
            "mlpipeline-ui-metadata": "/mlpipeline-ui-metadata.json",
            "accuracy": "/accuracy.json",
            "mlpipeline_metrics": "/mlpipeline-metrics.json"
        }
    ).set_display_name('analysis').after(train_faiss)

    baseline = 0.8
    with dsl.Condition(analysis.outputs["accuracy"] > baseline) as check_deploy:
        vop_deploy_model = dsl.VolumeOp(
            resource_name ="deploy-model-pvc"
            name="pvc_model",
            storage_class="",
            modes=dsl.VOLUME_MODE_RWM,
            size="5Gi",
            volume_name="deploy-model-pv"
        )

        deploy = dsl.ContainerOp(
            name="deploy mar",
            image="byeongjokim/mnist-deploy:latest",
            pvolumes={"/model": vop_model.volume, "/deploy-model": vop_deploy_model.volume}
        ).set_display_name('deploy').after(analysis)
        # .add_volume(k8s_client.V1Volume(name='model', host_path=k8s_client.V1HostPathVolumeSource(path='/model')))\
        # .add_volume_mount(k8s_client.V1VolumeMount(mount_path='/model', name='model'))\
        # .add_volume(k8s_client.V1Volume(name='deploy-model', host_path=k8s_client.V1HostPathVolumeSource(path='/deploy-model')))\
        # .add_volume_mount(k8s_client.V1VolumeMount(mount_path='/deploy-model', name='deploy-model'))\
        
        # .add_volume(k8s_client.V1Volume(name='model-pvc', persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(claim_name="model-pv-claim")))\
        # .add_volume_mount(k8s_client.V1VolumeMount(mount_path="/deploy-model", name="model-pvc"))\

if __name__=="__main__":
    host = "http://220.116.228.93:8080/pipeline"
    namespace = "kbj"
    experiment_name = "Mnist"
    pipeline_package_path = "pipeline.zip"
    run_name = "from collecting data to analysis"

    client = kfp.Client(host=host, namespace=namespace)
    kfp.compiler.Compiler().compile(mnist_pipeline, pipeline_package_path)
    experiment = client.create_experiment(name=experiment_name)
    run = client.run_pipeline(experiment.id, run_name, pipeline_package_path)