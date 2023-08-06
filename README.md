# rethge_torch
This repo contains help functions to convenient pytorch using. including plot, model train &amp; evaluate, results analysis and more...

* Those functions are using for computer vision(image classification)
* Those functions were wrote during my learning of pytorch, it can be directly use during DeepLearning project development, and also using for learning purpose
* Will continuously updating

some of funcs were adapted from this awesome tutorial: https://www.learnpytorch.io/ by Daniel Bourke, which is high recommend to check out

## install
`pip install rethge-torch`

## usage
`from rethge_torch import rethge`

  rethge.py includes help functions with:
  1. model related funcions (train/eval/test)
  2. Utils related funcions (set seeds, choose GPU/CPU, clean CUDA cache)
  3. File manipulate related functions
  4. Plot related functions (loss curve, learning rate, image display...)
  5. Data load related functions
  6. Result analyze/saving functions

### quick setup

choose devices:

`device = rethge.device_picking()`

config dataset transformation:

    from torchvision import transforms
    transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.TrivialAugmentWide(num_magnitude_bins=8),
        transforms.ToTensor()
    ])

get everything setup quickly:

     setup_dict = rethge.general_train_setup(Model=your_model, 
                        train_path=train_data_dir,
                        valid_path=valid_data_dir,
                        test_path=test_data_dir, # optional, if you don't have a test set, set to none
                        transform=transforms, 
                        test_transform = test_transforms, # if no test set, this is also set to none
                        batch_size=8, num_worker=8, epochs=100
                        )

'setup_dict' is a dict that contain dataloader, lr_scheduler(if needed), loss_fn, optimizing_func, classnames...
and training will be like:

     results = rethge.train_test_loop_with_amp(Model=your_model, 
                                train_loader=setup_dict['train_dataloader'],
                                test_loader=setup_dict['valid_dataloader'], 
                                epochs=100, optima=setup_dict['optima'], 
                                scheduler=setup_dict['scheduler'], 
                                loss_fn=setup_dict['loss_fn'], 
                                device=device,) 

## plot results

    rethge.plot_lr(results)
    rethge.plot_loss_curves(results)  

or you can also pass a csv file into 'plot_loss_curves'

    rethge.plot_loss_curves('results_path/xxxx.csv') 

## evaluate model

    loss, acc, predictions_tensor = rethge.eval_model(your_model, 
                                                      setup_dict['test_dataloader'],
                                                      setup_dict['loss_fn'],
                                                      device=device)

'pred_wrong_and_store' will loop through every single image in a folder and make a prediction, then gather all the results into a dataframe
you can set 'show=True' and 'n=5', it will display the top 5 most-wrong predictions

    pred_df = rethge.pred_wrong_and_store(path=test_path,
                                          Model=your_model,
                                          transform=transforms,
                                          class_names=setup_dict['class_name'],
                                          n = 5
                                          show = True
                                          device=device)

## result saving

    rethge.save_results(results, 'saving_path/xxxx.csv')
    rethge.save_model(model, 'saving_path/...', 'xxxx.pt') # or .pth
    
