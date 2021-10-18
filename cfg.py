# The using GPU ID
gpu = '0'  # '0,1,2,3'
finetune = False  # False True
resume = False  # False True
resume_path = './ckpt/saved_models/20.pth'

### TRAINING PARAMETERS ###
max_epoch = 30
batch_size = 10
val_batch = 4
data_shape = [128, 128*5]

### LOSS PARAMETERS ###
dice_coef = 10.0
valid_coef = 1.0
hole_coef = 6.0
tv_coef = 0.1
perc_coef = 0.05
style_coef = 100.0
# total variation calcuration method (mean or sum)
tv_loss = 'mean'

### OPTIMIZATION PARAMETERS ###
optim = 'Adam'
initial_lr = 0.0002
finetune_lr = 0.00005

### LOG INTERVALS ###
ckpt_dir = './ckpt'
temp_dir = './ckpt/_SCUT-EnsText_temp'

save_model_interval = 1
log_interval = 500

### DIRECTORY PATH ###
train_data_root = './train_set'
