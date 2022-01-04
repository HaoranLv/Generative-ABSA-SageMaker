from tqdm import tqdm
import torch
from eval_utils import *
from models import *


def predict(data,tokenizer,model,n_gpu, max_seq_length):
    """do predict"""
    device = torch.device(f'cuda:{n_gpu}')
    model.model.to(device)
    model.model.eval()
    inputs = tokenizer(
              data, max_length=max_seq_length, pad_to_max_length=True, truncation=True,
              return_tensors="pt",
            )
    outs = model.model.generate(input_ids=inputs["input_ids"].to(device), 
                                    attention_mask=inputs["attention_mask"].to(device), 
                                    max_length=1024)
    print(outs[0])
    dec=tokenizer.decode(outs[0], skip_special_tokens=True)
    

    return dec

def evaluate(tokenizer, data_loader, model, paradigm, n_gpu, task, sents, hasidx=False):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device(f'cuda:{n_gpu}')
    model.model.to(device)

    model.model.eval()
    outputs, targets = [], []
    for batch in tqdm(data_loader):
        # need to push the data to device
        #         print(batch['source_ids'])
        #         print(batch["target_ids"])
        #         print(batch["source_mask"])
        #         print(batch["target_mask"])
        outs = model.model.generate(input_ids=batch['source_ids'].to(device),
                                    attention_mask=batch['source_mask'].to(device),
                                    max_length=512)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
        #         print(dec)
        #         print(target)
        outputs.extend(dec)
        targets.extend(target)

    raw_scores, fixed_scores, all_labels, all_preds, all_preds_fixed = compute_scores(outputs, targets, sents, paradigm,
                                                                                      task,hasidx)
    results = {'raw_scores': raw_scores, 'fixed_scores': fixed_scores, 'labels': all_labels,
               'preds': all_preds, 'preds_fixed': all_preds_fixed}
    # pickle.dump(results, open(f"{args.output_dir}/results-{args.task}-{args.dataset}-{args.paradigm}.pickle", 'wb'))

    return raw_scores, fixed_scores
