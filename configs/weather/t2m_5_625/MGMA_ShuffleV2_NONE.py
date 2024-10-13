method = 'MGMA'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'shufflev2'
hid_S = 64
hid_T = 128
N_T = 8
N_S = 2
# training
lr = 5e-3
batch_size = 16
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 0
# MGMA
mgma_type = 'NONE'
mgma_num_groups = 8
block_type = 'i3d'
middle_ratio = 4