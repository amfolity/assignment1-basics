import torch
import wandb
import numpy as np
import torch.optim as optim
from .adapters import get_adamw_cls, run_cross_entropy, run_gradient_clipping, run_get_batch
from .utils import softmax, cross_entropy, Adamw, learning_rate_schedule, gradient_clipping, save_checkpoint, load_checkpoint
from .data_loader import data_loading
from torch.optim.lr_scheduler import LambdaLR

project_name = "llm-from-scratch-anna"


def training_together(model, config, path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=torch.float32)

    optimizer = Adamw(model.parameters(), config['lr'])
    total_train_loss = 0

    n_samples_train = 8985 ### where to place it?? is it needed??
    n_samples_val = 999

    train = np.memmap('train.bin', dtype='int64', mode='r', shape=(n_samples_train,))
    val = np.memmap('val.bin', dtype='int64', mode='r', shape=(n_samples_val,))

    #iterations_per_epoch = len(dataset) // config['context_length']
    lr_schema = lambda step : learning_rate_schedule(it=step,
                            max_learning_rate=1.0,
                            min_learning_rate=0.01,
                            warmup_iters=100,
                            cosine_cycle_iters=200)
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schema)

    with wandb.init(project=project_name, config=config) as run:

        model.train()
        
        for itt in range(config['steps']):
            
            batch_in, batch_target = data_loading(train, config['batch_size'], config['context_length'], device)
            
            optimizer.zero_grad()
            
            outputs = model(batch_in)
            
            loss = run_cross_entropy(outputs.view(-1, outputs.size(-1)), batch_target.view(-1))
            loss.backward()
            
            gradient_clipping(model.parameters(), max_l2_norm=1.0) ## what is max_l2_norm?

            optimizer.step()                 
            
            run.log({'lr':scheduler.get_last_lr()[0], 'train_loss':loss.item()}) 
            #run.log({'lr':config['lr'], 'batch_loss':loss.item()}) 
            scheduler.step()
            
            save_checkpoint(model=model, optimizer=optimizer, iteration=itt, out=path+f'iteration{itt}.pth') 

        model.eval()
        with torch.no_grad():
            val_batch_in, val_batch_target = data_loading(val, config['batch_size'], config['context_length'], device)
            val_outputs = model(val_batch_in)
            val_loss = run_cross_entropy(val_outputs.view(-1, val_outputs.size(-1)), val_batch_target.view(-1))
            run.log({
            'val_loss': val_loss.item()
            })
    run.finish()
        
            
    
    