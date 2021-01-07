import os
import argparse

def main(args):
    if not os.path.isdir(args.export_path):
        os.mkdir(args.export_path)
        print('[+] Mkdir export path:', args.export_path)

    # cmd = "torch-model-archiver --model-name embedding --version 1.0 --serialized-file model.pt --extra-files MyHandler.py,faiss_index.bin,faiss_label.json --handler handler.py"
    cmd = "torch-model-archiver "
    cmd += "--model-name {} ".format(args.model_name)
    cmd += "--version {} ".format(args.version)
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

    config_file = os.path.join(args.config_path, "config.properties")
    cmd = "cp ./config.properties {}".format(config_file)
    print(cmd)
    os.system(cmd)
    
    # git add yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', type=str, default="embedding")
    parser.add_argument('--version', type=str, default="1.0")
    
    parser.add_argument('--model_dir', type=str, default='/model')
    parser.add_argument('--model_file', type=str, default='model.pt')
    parser.add_argument('--faiss_model_file', type=str, default='faiss_index.bin')
    parser.add_argument('--faiss_label_file', type=str, default='faiss_label.json')

    parser.add_argument('--handler_class', type=str, default="MyHandler.py")
    
    parser.add_argument('--handler', type=str, default="handler.py")

    parser.add_argument('--export_path', type=str, default='/deploy-model/model-store')
    parser.add_argument('--config_path', type=str, default='/deploy-model/config')

    args = parser.parse_args()

    main(args)
    