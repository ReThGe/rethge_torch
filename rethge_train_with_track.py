import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from typing import Tuple, Dict, List
from timeit import default_timer as timer
from tqdm.auto import tqdm
import os, gc
from datetime import datetime

from rethge_torch import print_train_time, test_step, save_results


def train_test_loop_with_amp_and_tracker(Model: torch.nn.Module, 
          train_loader: torch.utils.data.DataLoader, 
          test_loader: torch.utils.data.DataLoader, 
          optima: torch.optim.Optimizer,
          epochs: int,
          writer: torch.utils.tensorboard.SummaryWriter = None, 
          device: torch.device = torch.device("cpu"),
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          scheduler: torch.optim.lr_scheduler = None) -> Dict[str, List]:
    
    """train model with AMP and a tracker of tensorborad"""

    if scheduler is not None:
        results = {'train_loss': [],
                'train_acc': [],
                'test_loss': [],
                'test_acc': [],
                'learning rate': []}
    else:
        results = {'train_loss': [],
                'train_acc': [],
                'test_loss': [],
                'test_acc': [],}


    # train_loss, train_acc = 0, 0

    Model.to(device)
    Model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    time_start = timer()
    for ep in tqdm(range(epochs)):

        train_loss, train_acc = 0, 0 #?? maybe to avoid nan?

        for X, y in train_loader:     
            X, y = X.to(device), y.to(device)

            optima.zero_grad(set_to_none=True)
            # for param in Model.parameters():
            #     param.grad = None

            with torch.autocast(device_type=str(device), dtype=torch.float16):

                y_pred_t = Model(X) 
                loss_t = loss_fn(y_pred_t, y)
                
            # or maybe we should move this two line inside of AMP block? 
            train_loss += loss_t.item() # .item() turn single tensor into a single scaler
            y_pred_t_class = torch.argmax(y_pred_t, dim=1)
            train_acc += torch.eq(y_pred_t_class, y).sum().item()/len(y_pred_t) * 100

            scaler.scale(loss_t).backward() # none type
            
            scaler.unscale_(optima)
     
            torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=0.1)
        
            scaler.step(optima)  
            scaler.update()

            # loss_t.backward()
            # optima.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        if train_acc > 100:
            train_acc = 100.0000

        test_acc, test_loss = test_step(Model=Model,
                    data_loader=test_loader,
                    loss_fn=loss_fn,
                    device=device)
        
        if scheduler is not None:
            optima.zero_grad(set_to_none=True)
            optima.step()
            current_lr = optima.param_groups[0]['lr']
            results['learning rate'].append(current_lr)
            scheduler.step()
        
        print(f"Epoch: {ep+1} | "
                f"train_loss: {train_loss:.4f} | "  # nan???
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "    # nan???
                f"test_acc: {test_acc:.4f}"
                )
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

    
        ### New: Experiment tracking ###
        # Add loss results to SummaryWriter
        if writer:
            writer.add_scalars(main_tag="Loss", 
                                tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                                                global_step=ep)

            # Add accuracy results to SummaryWriter
            writer.add_scalars(main_tag="Accuracy", 
                            tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc}, 
                                                global_step=ep)
            
            # Track the PyTorch model architecture
            # writer.add_graph(model=Model, 
            #                 # Pass in an example input
            #                 input_to_model=torch.randn(batch_size, 3, res, res).to(device))


        else:
            pass
    # Close the writer
    writer.close()

    gc.collect()
    torch.cuda.empty_cache()

    time_end = timer()
    print_train_time(start=time_start,
                        end=time_end,
                        device=device)

    # Return the filled results at the end of the epochs
    return results




def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    """
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)