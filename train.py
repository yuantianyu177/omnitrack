import os
import random
import shutil
import yaml
import tqdm
import numpy as np
import torch
import torch.utils.data
from config import config_parser
from tensorboardX import SummaryWriter
from dataset.create_dataset import get_training_dataset
import time
from eval import eval_one_step
from trainer.combo_trainer import ComboTrainer

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(args):
    seq_name = os.path.basename(args.data_dir.rstrip('/'))
    now = time.strftime("%y%m%d-%H%M", time.localtime())
    out_dir = os.path.join(args.save_dir, f"{now}_{args.expname}_{seq_name}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"optimizing for {seq_name}...\n output is saved in {out_dir}")
    log_dir = f"logs/{now}_{args.expname}_{seq_name}"
    writer = SummaryWriter(log_dir)
    args.out_dir = out_dir

    # save the args and config files
    f = os.path.join(out_dir, 'args.yaml')
    with open(f, 'w') as file:
        filtered_args = {arg: getattr(args, arg) for arg in sorted(vars(args)) if not arg.startswith('_')}
        yaml.dump(filtered_args, file, default_flow_style=False, sort_keys=True)

    if args.config:
        f = os.path.join(out_dir, 'config.yaml')
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    g = torch.Generator()
    g.manual_seed(args.loader_seed)

    dataset, data_sampler = get_training_dataset(args, max_interval=args.start_interval)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.num_pairs,
                                              worker_init_fn=seed_worker,
                                              generator=g,
                                              num_workers=args.num_workers,
                                              sampler=data_sampler,
                                              shuffle=True if data_sampler is None else False,
                                              pin_memory=True)

    if args.trainer == 'combo':
        trainer = ComboTrainer(args)
    else:
        raise ValueError(f"Invalid trainer type: {args.trainer}")

    eval_it = 500
    start_step = trainer.step + 1
    step = start_step
    run_time_acc = 0    # accumulated run time excluding eval time
    run_time_st = time.time()   # end time of last eval

    pbar = tqdm(range(start_step, args.num_iters + start_step + 1), desc="Training", unit="step")
    torch.cuda.empty_cache()
    for step in pbar:
        for batch in data_loader:
            # training
            loss = trainer.train_one_step(step, batch)
            pbar.set_postfix(loss=f"{loss:.2f}", step=step)

            # evaluation
            if step % eval_it == 0 and trainer.eval:
                run_time_acc += time.time() - run_time_st
                trainer.deform_nvp.eval()
                with torch.no_grad():
                    res = eval_one_step(trainer, step, depth_err=0.04)
                res['run_time'] = run_time_acc
                res['step'] = step
                np.save(os.path.join(out_dir, 'eval', f'metric_{step:08d}.npy'), res)
                trainer.deform_nvp.train()
                run_time_st = time.time()

            # log
            if step % 100 == 0:
                trainer.log(writer, step)
                dataset.increase_range()



if __name__ == '__main__':
    args = config_parser()
    train(args)
