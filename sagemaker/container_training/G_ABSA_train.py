from argparse import ArgumentParser
import os
from mmcv import Config
from mmcv.runner import set_random_seed
import json
import subprocess
import sys
import shutil

# if "MMACTION2" not in os.environ:
#     os.environ["MMACTION2"] = "/opt/ml/code/mmaction2"
# if "SM_MODEL_DIR" not in os.environ:
#     os.environ["SM_MODEL_DIR"] = "/opt/ml/model"
# if "SM_OUTPUT_DATA_DIR" not in os.environ:
#     os.environ["SM_OUTPUT_DATA_DIR"] = "/opt/ml/output"
# if "SM_CHANNEL_TRAINING" not in os.environ:
#     os.environ["SM_CHANNEL_TRAINING"] = '/opt/ml/input/data/training'
# if "SM_NUM_GPUS" not in os.environ:
#     os.environ["SM_NUM_GPUS"] = "1"
# if "SM_NUM_CPUS" not in os.environ:
#     os.environ["SM_NUM_CPUS"] = "32"
# if "SM_HOSTS" not in os.environ:
#     os.environ["SM_HOSTS"] = "[\"algo-1\"]"
# if "SM_CURRENT_HOST" not in os.environ:
#     os.environ["SM_CURRENT_HOST"] = "algo-1"

def get_training_world():

    """
    Calculates number of devices in Sagemaker distributed cluster
    """

    # Get params of Sagemaker distributed cluster from predefined env variables
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_cpus = int(os.environ["SM_NUM_CPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]

    # Define PyTorch training world
    world = {}
    world["number_of_processes"] = num_gpus if num_gpus > 0 else num_cpus
    world["number_of_machines"] = len(hosts)
    world["size"] = world["number_of_processes"] * world["number_of_machines"]
    world["machine_rank"] = hosts.index(current_host)
    world["master_addr"] = hosts[0]
    world["master_port"] = "55555" # port is defined by Sagemaker

    return world
def init_args():
    parser = ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='tasd-cn', type=str, required=True,
                        help="The name of the task, selected from: [uabsa, aste, tasd, aope]")
    parser.add_argument("--nodes", default=1, type=int, required=True,
                        help="The name of the task, selected from: [uabsa, aste, tasd, aope]")
    parser.add_argument("--data_root", default=os.environ["SM_CHANNEL_TRAINING"], type=str, required=True,
                        help="The path of data root")
    parser.add_argument("--ckpoint_path", default='./outputs/tasd-cn/ctrip/extraction/cktepoch=1.ckpt', type=str, required=False)
    parser.add_argument("--text", default='早餐一般般，勉勉强强填饱肚子，样式可选性不多，可能是疫情的影响吧。不过酒店的服务不错，五个小孩早餐都送了，点👍。由于酒店历史有点长，所以设施感觉一般般，整体还可以，三钻吧', type=str, required=False)
    parser.add_argument("--dataset", default='rest16', type=str, required=True,
                        help="The name of the dataset, selected from: [laptop14, rest14, rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='lemon234071/t5-base-Chinese', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--save_strategy", default='epoch', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--save_total_limit", default=5, type=int,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--paradigm", default='extraction', type=str, required=True,
                        help="The way to construct target sentence, selected from: [annotation, extraction]")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true',
                        help="Whether to run direct eval on the dev/test set.")
    parser.add_argument("--do_direct_predict", action='store_true', 
                        help="Whether to run direct eval on the dev/test set.")

    # Other parameters
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    args, unknown = parser.parse_known_args()

    # set up output dir which looks like './aste/rest14/extraction/'
    # if not os.path.exists('./outputs'):
    #     os.mkdir('./outputs')
    
    # task_dir = f"./outputs/{args.task}"
    task_dir = os.path.join(os.environ["SM_OUTPUT_DATA_DIR"], f"{args.task}")
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    # task_dataset_dir = f"{task_dir}/{args.dataset}"
    task_dataset_dir = os.path.join(task_dir, f"{args.dataset}")
    if not os.path.exists(task_dataset_dir):
        os.mkdir(task_dataset_dir)

    # output_dir = f"{task_dataset_dir}/{args.paradigm}"
    output_dir = os.path.join(task_dataset_dir,f"{args.paradigm}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir


    return args, unknown
def training_configurator(args, world):
    
    """
    Configure training process by updating config file: 
    - takes base config from MMAction2 templates;
    - updates it with SageMaker specific data locations;
    - overrides with user-defined options.
    """
    
    # updating path to config file inside SM container
    abs_config_path = os.path.join(os.environ["MMACTION2"], args.config_file)
    cfg = Config.fromfile(abs_config_path)
    
    if args.dataset.lower() == "kinetics400_tiny":
        root_dir = os.environ["SM_CHANNEL_TRAINING"] # By default, data will be download to /opt/ml/input/data/training
        cfg.dataset_type = 'VideoDataset'
        cfg.data_root = os.path.join(root_dir, 'train/')
        cfg.data_root_val = os.path.join(root_dir, 'val/')
        cfg.ann_file_train = os.path.join(root_dir, 'kinetics_tiny_train_video.txt')
        cfg.ann_file_val = os.path.join(root_dir, 'kinetics_tiny_val_video.txt')
        cfg.ann_file_test = 'kinetics400_tiny/kinetics_tiny_val_video.txt'

        cfg.data.test.type = 'VideoDataset'
        cfg.data.test.ann_file = os.path.join(root_dir, 'kinetics_tiny_val_video.txt')
        cfg.data.test.data_prefix = os.path.join(root_dir, 'val/')

        cfg.data.train.type = 'VideoDataset'
        cfg.data.train.ann_file = os.path.join(root_dir, 'kinetics_tiny_train_video.txt')
        cfg.data.train.data_prefix = os.path.join(root_dir, 'train/')

        cfg.data.val.type = 'VideoDataset'
        cfg.data.val.ann_file = os.path.join(root_dir, 'kinetics_tiny_val_video.txt')
        cfg.data.val.data_prefix = os.path.join(root_dir, 'val/')

        # The flag is used to determine whether it is omnisource training
        cfg.setdefault('omnisource', False)
        # Modify num classes of the model in cls_head
        cfg.model.cls_head.num_classes = 2
        # We can use the pre-trained TSN model
#         cfg.load_from = os.path.join(os.environ["MMACTION2"], './checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth')  # TODO need to use a pretrained model

        # Set up working dir to save files and logs.
        cfg.work_dir = './tutorial_exps'

        # The original learning rate (LR) is set for 8-GPU training.
        # We divide it by 8 since we only use one GPU.
        cfg.data.videos_per_gpu = cfg.data.videos_per_gpu // 16
        cfg.optimizer.lr = cfg.optimizer.lr / 8 / 16
        cfg.total_epochs = 30

        # We can set the checkpoint saving interval to reduce the storage cost
        cfg.checkpoint_config.interval = 10
        # We can set the log print interval to reduce the the times of printing log
        cfg.log_config.interval = 5

        # Set seed thus the results are more reproducible
        cfg.seed = 0
        set_random_seed(0, deterministic=False)
        cfg.gpu_ids = range(1)

        # Overriding config with options
        if args.options is not None:
            cfg.merge_from_dict(args.options)
        
        # scaling LR based on number of training processes
        if args.auto_scale:
            cfg = auto_scale_config(cfg, world)
        
        updated_config = os.path.join(os.getcwd(), "updated_config.py")
        cfg.dump(updated_config)

        # We can initialize the logger for training and have a look
        # at the final config used for training
        print(f'Config:\n{cfg.pretty_text}')
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented.\
                                    Currently only kinetics400_tiny-style datasets are available.")
              
    return updated_config


def auto_scale_config(cfg, world):
    """
    Method automatically scales learning rate
    based on number of processes in distributed cluster.
    
    When scaling, we take user-provided config as a config for single node with 8 GPUs
    and scale it based on total number of training processes.
    
    Note, that batch size is not scaled, as MMDetection uses relative
    batch size: cfg.data.samples_per_gpu
    """
    
    old_world_size = 8 # Note, this is a hardcoded value, as MMDetection configs are build for single 8-GPU V100 node.
    old_lr = cfg.optimizer.lr
    old_lr_warmup = cfg.lr_config.warmup_iters
    scale = world["size"] / old_world_size
    
    cfg.optimizer.lr = old_lr * scale
    cfg.lr_config.warmup_iters = old_lr_warmup / scale
    
    print(f"""Initial learning rate {old_lr} and warmup {old_lr_warmup} were scaled \
          to {cfg.optimizer.lr} and {cfg.lr_config.warmup_iters} respectively.
          Each GPU has batch size of {cfg.data.samples_per_gpu},
          Total number of GPUs in training cluster is {world['size']}.
          Effective batch size is {cfg.data.samples_per_gpu * world['size']}""")
    
    return cfg

def options_to_dict(options):
    """
    Takes string of options in format of 'key1=value1; key2=value2 ...'
    and produces dictionary object {'key1': 'value1', 'key2':'value2'...}.
    
    It also supports lists of values: key3=v1,v2,v3.
    """
    
    options_dict = dict(item.split("=") for item in options.split("; ")) 
    
    for key, value in options_dict.items():
        value = [_parse_int_float_bool(v) for v in value.split(",")]
        if len(value) == 1:
            value = value[0]
        options_dict[key] = value
    return options_dict


def _parse_int_float_bool(val):
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    if val.lower() in ['true', 'false']:
        return True if val.lower() == 'true' else False
    return val


def save_model(work_dir, model_dir):
    """
    This method copies model trained weights and config 
    from output directory to model directory.
    Sagemaker then automatically archives content of model directory
    and adds it to model registry once training job is completed.
    """
    

    # Copy checkpoint file
    last_model=None
    flag=0
    for file in os.listdir(work_dir):
        if file.endswith(".ckpt"):
            if int(file[(file.index('=')+1):-5])>=flag:
                flag=int(file[(file.index('=')+1):-5])
                last_model=file
    try:
        checkpoint_path = os.path.join(work_dir, last_model)
        new_checkpoint_path = os.path.join(model_dir, file)
        shutil.copyfile(checkpoint_path, new_checkpoint_path)
    except Exception as e:
        print(f"Exception when trying to copy {checkpoint_path} to {new_checkpoint_path}.")
        print(e)
    
    print(f"Model config and checkpoints are saved to {model_dir}.")


if __name__ == "__main__":
    
    # Get initial configuration to select appropriate HuggingFace task and its configuration
    print('Starting training...')
    args, unknown= init_args()
    # parser = ArgumentParser()
    # parser.add_argument('--config-file', type=str, default=None, metavar="FILE", 
    #                     help="Only default MMAction2 configs are supported now. \
    #                     See for details: https://github.com/open-mmlab/mmaction2/tree/master/configs/")
    # parser.add_argument('--dataset', type=str, default="kinetics400_tiny", help="Define which dataset format to use.")
    # parser.add_argument('--options', nargs='+', type=str, default=None, help='Config overrides.')
    # parser.add_argument('--auto-scale', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], 
    #                     default=False, help="whether to scale batch parameters and learning rate based on cluster size")
    # parser.add_argument('--validate', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], 
    #                 default=False, help="whether to scale batch parameters and learning rate based on cluster size")
   
    
    if unknown:
        print(f"Following arguments were not recognized and won't be used: {unknown}")

    # Derive parameters of distributed training cluster in Sagemaker
    world = get_training_world()  

    # Update config file
    # config_file = training_configurator(args, world)
              
    # Train script config
    launch_config = [ "MASTER_ADDR={}".format(world['master_addr']), 
                     "MASTER_PORT={}".format(world['master_port']), 
                     "WORLD_SIZE={}".format(world["size"]), 
                    ]
 
    train_config = ['python', os.path.join(os.environ["G_ABSA"], "main.py"), 
                    "--dataset", args.dataset,
                    "--task", args.task,
                    "--model_name_or_path", args.model_name_or_path,
                    "--n_gpu", args.n_gpu,
                    "--do_train",
                    "--train_batch_size",args.train_batch_size,
                    "--gradient_accumulation_steps",args.gradient_accumulation_steps,
                    "--eval_batch_size",args.eval_batch_size,
                    "--learning_rate", args.learning_rate,
                    "--num_train_epochs", args.num_train_epochs,
                    "--nodes", args.nodes]
    

    # Concat Pytorch Distributed Launch config and MMaction2 config
    joint_cmd = " ".join(str(x) for x in launch_config+train_config)
    # joint_cmd = " ".join(str(x) for x in train_config)
    print("Following command will be executed: \n", joint_cmd)
    
    process = subprocess.Popen(joint_cmd,  stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    
    while True:
        output = process.stdout.readline()
        
        if process.poll() is not None:
            break
        if output:
            print(output.decode("utf-8").strip())
    rc = process.poll()
    
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=joint_cmd)
    
    # Before completing training, saving model artifacts
    save_model(args.output_dir, os.environ['SM_MODEL_DIR'])
    
    sys.exit(process.returncode)
    
