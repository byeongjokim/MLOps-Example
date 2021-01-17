import os
import argparse
from datetime import datetime
from kubernetes import client, config
import yaml
import requests

def archive(args, version):
    model_name_version = args.model_name+"_"+version

    if not os.path.isdir(args.export_path):
        os.mkdir(args.export_path)
        print('[+] Mkdir export path:', args.export_path)
    
    # cmd = "torch-model-archiver --model-name embedding --version 1.0 --serialized-file model.pt --extra-files MyHandler.py,faiss_index.bin,faiss_label.json --handler handler.py"
    cmd = "torch-model-archiver "
    cmd += "--model-name {} ".format(model_name_version)
    cmd += "--version {} ".format(version)
    cmd += "--serialized-file {} ".format(os.path.join(args.model_dir, args.model_file))

    extra_files = [
        args.handler_class,
        os.path.join(args.model_dir, args.faiss_model_file),
        os.path.join(args.model_dir, args.faiss_label_file)
    ]
    cmd += "--extra-files {} ".format(",".join(extra_files))
    cmd += "--handler {} ".format(args.handler)
    cmd += "--export-path {} ".format(args.export_path)
    cmd += "-f"
    print(cmd)
    os.system(cmd)

    if not os.path.isdir(args.config_path):
        os.mkdir(args.config_path)
        print('[+] Mkdir config path:', args.config_path)
    
    config="""inference_address=http://0.0.0.0:{}
    management_address=http://0.0.0.0:{}
    metrics_address=http://0.0.0.0:{}
    job_queue_size=100
    load_models=standalone""".format(args.pred_port, args.manage_port, args.metric_port)

    config_file = os.path.join(args.config_path, "config.properties")

    with open(config_file, "w") as f:
        f.write(config)

def serving(args, version):
    model_name_version = args.model_name+"_"+version
    config.load_incluster_config()

    k8s_apps_v1 = client.AppsV1Api()
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(
            labels={"app":"torchserve"}
        ),
        spec=client.V1PodSpec(
            volumes=[
                client.V1Volume(
                    name="persistent-storage",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name="serving-model-pvc"
                    )
                )
            ],
            containers=[
                client.V1Container(
                    name="torchserve",
                    image="byeongjokim/torchserve",
                    args=["torchserve", "--start",  "--model-store", "/home/model-server/shared/model-store/", "--ts-config", "/home/model-server/shared/config/config.properties"],
                    image_pull_policy="Always",
                    ports=[
                        client.V1ContainerPort(
                            name="ts",
                            container_port=args.pred_port
                        ),
                        client.V1ContainerPort(
                            name="ts-management",
                            container_port=args.manage_port
                        ),
                        client.V1ContainerPort(
                            name="ts-metrics",
                            container_port=args.metric_port
                        )
                    ],
                    volume_mounts=[
                        client.V1VolumeMount(
                            name="persistent-storage",
                            mount_path="/home/model-server/shared/"
                        )
                    ],
                    resources=client.V1ResourceRequirements(
                        limits={
                            "cpu":1,
                            "memory":"4Gi",
                            "nvidia.com/gpu": 0
                        },
                        requests={
                            "cpu":1,
                            "memory":"1Gi",
                        }
                    )
                )
            ]
        )
    )
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(
            name="torchserve",
            labels={
                "app":"torchserve",
                "app.kubernetes.io/version":version
            }
        ),
        spec=client.V1DeploymentSpec(
            replicas=2,
            selector=client.V1LabelSelector(
                match_labels={"app":"torchserve"}
            ),
            strategy=client.V1DeploymentStrategy(
                type="RollingUpdate",
                rolling_update=client.V1RollingUpdateDeployment(
                    max_surge=1,
                    max_unavailable=1,
                )
            ),
            template=template
        )
    )
    
    k8s_core_v1 = client.CoreV1Api()
    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(
            name="torchserve",
            labels={
                "app":"torchserve",
                "app.kubernetes.io/version":version
            }
        ),
        spec=client.V1ServiceSpec(
            type="LoadBalancer",
            selector={"app":"torchserve"},
            ports=[
                client.V1ServicePort(
                    name="preds",
                    port=args.pred_port,
                    target_port="ts"
                ),
                client.V1ServicePort(
                    name="mdl",
                    port=args.manage_port,
                    target_port="ts-management"
                ),
                client.V1ServicePort(
                    name="metrics",
                    port=args.metric_port,
                    target_port="ts-metrics"
                )
            ]
        )
    )

    try:
        k8s_apps_v1.create_namespaced_deployment(body=deployment, namespace="kbj")
        print("[+] Deployment created")
        k8s_core_v1.create_namespaced_service(body=service, namespace="kbj")
        print("[+] Service created")
        #portforward

    except Exception as ex:
        #check url
        print("========================")
        print(ex)
        print("========================")
    
    # cmd= 'curl -v -X POST "http://torchserve:{}/models?model_name={}&url={}.mar"'.format(args.manage_port, args.model_name, model_name_version)
    
    url = "http://torchserve:{}/models?model_name={}&url={}.mar".format(args.manage_port, args.model_name, model_name_version)
    res = requests.post(url)
    print(res.text)

def main(args):
    now = datetime.now()
    version = now.strftime("%y%m%d_%H%M")

    archive(args, version)
    serving(args, version)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', type=str, default="embedding")
    
    parser.add_argument('--model_dir', type=str, default='/model')
    parser.add_argument('--model_file', type=str, default='model.pt')
    parser.add_argument('--faiss_model_file', type=str, default='faiss_index.bin')
    parser.add_argument('--faiss_label_file', type=str, default='faiss_label.json')

    parser.add_argument('--handler_class', type=str, default="MyHandler.py")
    
    parser.add_argument('--handler', type=str, default="handler.py")

    parser.add_argument('--export_path', type=str, default='/deploy-model/model-store')
    parser.add_argument('--config_path', type=str, default='/deploy-model/config')

    parser.add_argument('--pred_port', type=int, default=8082)
    parser.add_argument('--manage_port', type=int, default=8083)
    parser.add_argument('--metric_port', type=int, default=8084)

    args = parser.parse_args()

    main(args)
    