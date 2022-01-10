import argparse
import os
import logging
import time
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from transformers import  T5Tokenizer
# from datasets_utils.data_utils import ABSADataset
from datasets_utils.data_utils import write_results_to_log, read_line_examples_from_file
from eval_utils import *
from models import *
from utils import *

logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='tasd-cn', type=str, required=True,
                        help="The name of the task, selected from: [uabsa, aste, tasd, aope]")
    parser.add_argument("--nodes", default=1, type=int, required=True,
                        help="The name of the task, selected from: [uabsa, aste, tasd, aope]")
    parser.add_argument("--data_root", default='./data', type=str, required=False,
                        help="The path of data root")
    parser.add_argument("--ckpoint_path", default='./outputs/tasd-cn/ctrip/extraction/cktepoch=1.ckpt', type=str, required=False)
    parser.add_argument("--text", default='早餐一般般，勉勉强强填饱肚子，样式可选性不多，可能是疫情的影响吧。不过酒店的服务不错，五个小孩早餐都送了，点👍。由于酒店历史有点长，所以设施感觉一般般，整体还可以，三钻吧', type=str, required=False)
    parser.add_argument("--dataset", default='ctrip', type=str, required=True,
                        help="The name of the dataset, selected from: [laptop14, rest14, rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='lemon234071/t5-base-Chinese', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--save_strategy", default='epoch', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--save_total_limit", default=5, type=int,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--paradigm", default='extraction', type=str, required=False,
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

    args = parser.parse_args()

    # set up output dir which looks like './aste/rest14/extraction/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    task_dir = f"./outputs/{args.task}"
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    task_dataset_dir = f"{task_dir}/{args.dataset}"
    if not os.path.exists(task_dataset_dir):
        os.mkdir(task_dataset_dir)

    output_dir = f"{task_dataset_dir}/{args.paradigm}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args

if __name__ == "__main__":
    # initialization
    args = init_args()
    print("\n", "=" * 30, f"NEW EXP: {args.task.upper()} on {args.dataset}", "=" * 30, "\n")

    seed_everything(args.seed)
    # print(args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    # show one sample to check the sanity of the code and the expected output
    print(f"Here is an example (from dev set) under `{args.paradigm}` paradigm:")
    dataset = ABSADataset(data_root=args.data_root,tokenizer=tokenizer, data_dir=args.dataset, data_type='dev',
                          paradigm=args.paradigm, task=args.task, max_len=args.max_seq_length)
    data_sample = dataset[2]  # a random data sample
    print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
    print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))

    # training process
    if args.do_train:
        print("\n****** Conduct Training ******")
        model = T5FineTuner(args)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=args.output_dir, prefix="ckt", monitor='val_loss', mode='min', save_top_k=30
        )

        # prepare for trainer
        train_params = dict(
            num_nodes=int(args.nodes),
            distributed_backend='ddp',
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=int(args.n_gpu),
            gradient_clip_val=1.0,
            # amp_level='O1',
            max_epochs=args.num_train_epochs,
            checkpoint_callback=checkpoint_callback,
            callbacks=[LoggingCallback(logger)],
        )

        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

        # save the final model
        model.model.save_pretrained(args.output_dir)

        print("Finish training and saving the model!")

    if args.do_eval:
        sents, _ = read_line_examples_from_file(f'data/{args.task}/{args.dataset}/test.txt')

        print("\n****** Conduct Evaluating ******")

        # model = T5FineTuner(args)
        dev_results, test_results = {}, {}
        best_f1, best_checkpoint, best_epoch = -999999.0, None, None
        all_checkpoints, all_epochs = [], []

        # retrieve all the saved checkpoints for model selection
        saved_model_dir = args.output_dir
        for f in os.listdir(saved_model_dir):
            file_name = os.path.join(saved_model_dir, f)
            if 'cktepoch' in file_name:
                all_checkpoints.append(file_name)

        # conduct some selection (or not)
        print(f"We will perform validation on the following checkpoints: {all_checkpoints}")

        # load dev and test datasets
        dev_dataset = ABSADataset(args.data_root, tokenizer, data_dir=args.dataset, data_type='dev',
                                  paradigm=args.paradigm, task=args.task, max_len=args.max_seq_length)
        dev_loader = DataLoader(dev_dataset, batch_size=32, num_workers=0)

        test_dataset = ABSADataset(args.data_root, tokenizer, data_dir=args.dataset, data_type='test',
                                   paradigm=args.paradigm, task=args.task, max_len=args.max_seq_length)
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)

        for checkpoint in all_checkpoints:
            epoch = checkpoint.split('=')[-1][:-5] if len(checkpoint) > 1 else ""
            # only perform evaluation at the specific epochs ("15-19")
            # eval_begin, eval_end = args.eval_begin_end.split('-')
            if 0 <= int(epoch) < 100:
                all_epochs.append(epoch)

                # reload the model and conduct inference
                print(f"\nLoad the trained model from {checkpoint}...")
                model_ckpt = torch.load(checkpoint)
                model = T5FineTuner(model_ckpt['hyper_parameters'])
                model.load_state_dict(model_ckpt['state_dict'])

                dev_result,fix_res = evaluate(tokenizer, dev_loader, model, args.n_gpu, args.paradigm, args.task, sents)
                if dev_result['f1'] > best_f1:
                    best_f1 = dev_result['f1']
                    best_checkpoint = checkpoint
                    best_epoch = epoch

                # add the global step to the name of these metrics for recording
                # 'f1' --> 'f1_1000'
                dev_result = dict((k + '_{}'.format(epoch), v) for k, v in dev_result.items())
                dev_results.update(dev_result)

                test_result = evaluate(tokenizer, test_loader, model, args.n_gpu, args.paradigm, args.task)
                test_result = dict((k + '_{}'.format(epoch), v) for k, v in test_result.items())
                test_results.update(test_result)

        # print test results over last few steps
        print(f"\n\nThe best checkpoint is {best_checkpoint}")
        best_step_metric = f"f1_{best_epoch}"
        print(f"F1 scores on test set: {test_results[best_step_metric]:.4f}")

        print("\n* Results *:  Dev  /  Test  \n")
        metric_names = ['f1', 'precision', 'recall']
        for epoch in all_epochs:
            print(f"Epoch-{epoch}:")
            for name in metric_names:
                name_step = f'{name}_{epoch}'
                print(f"{name:<10}: {dev_results[name_step]:.4f} / {test_results[name_step]:.4f}", sep='  ')
            print()

        results_log_dir = './results_log'
        if not os.path.exists(results_log_dir):
            os.mkdir(results_log_dir)
        log_file_path = f"{results_log_dir}/{args.task}-{args.dataset}.txt"
        write_results_to_log(log_file_path, test_results[best_step_metric], args, dev_results, test_results, all_epochs)

    # evaluation processcktepoch
    if args.do_direct_eval:
        print("\n****** Conduct Evaluating with the last state ******")
        device=torch.device(f'cuda:{args.n_gpu}')
        checkpoint=args.ckpoint_path
        print(f"\nLoad the trained model from {checkpoint}...")
        model_ckpt = torch.load(checkpoint,map_location=device)
        model = T5FineTuner(model_ckpt['hyper_parameters'])
        model.load_state_dict(model_ckpt['state_dict'])
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        # model = T5FineTuner(args)

        # print("Reload the model")
        # model.model.from_pretrained(args.output_dir)
        sents, _ = read_line_examples_from_file(os.path.join(args.data_dir,f'{args.task}/{args.dataset}/test.txt'))
        test_dataset = ABSADataset(args.data_root, tokenizer, data_dir=args.dataset, data_type='test',
                                   paradigm=args.paradigm, task=args.task, max_len=args.max_seq_length)
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)
#         print(test_loader.device)
        raw_scores, fixed_scores = evaluate(tokenizer, model, args.paradigm, args.n_gpu, args.task, sents, hasidx=False)
        # print(scores)

#         # write to file
#         log_file_path = f"results_log/{args.task}-{args.dataset}.txt"
#         local_time = time.asctime(time.localtime(time.time()))
#         exp_settings = f"{args.task} on {args.dataset} under {args.paradigm}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}"
#         exp_results = f"Raw F1 = {raw_scores['f1']:.4f}"
#         log_str = f'============================================================\n'
#         log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"
#         with open(log_file_path, "a+") as f:
#             f.write(log_str)
# prediction process
    if args.do_direct_predict:
        print("\n****** Conduct predicting with the last state ******")
        checkpoint=args.ckpoint_path
        print(f"\nLoad the trained model from {checkpoint}...")
        device=torch.device(f'cuda:{args.n_gpu}')
        model_ckpt = torch.load(checkpoint,map_location=device)
        model = T5FineTuner(model_ckpt['hyper_parameters'])
        model.load_state_dict(model_ckpt['state_dict'])
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    
        sents=[args.text]

        s=time.time()
        for i in sents:
    # # print(test_loader.device)
            pred = predict(i, tokenizer,model, args.n_gpu, args.max_seq_length)
            print('sents:',i)
            print('pred:',pred)
        e=time.time()
        print(e-s)