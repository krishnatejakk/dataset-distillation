# from cvar_pyutils.ccc import submit_job

# submit_job(command_to_run="python main.py --mode distill_basic --dataset MNIST --arch LeNet",
#         duration="12h",
#         num_nodes=1,
#         num_cores=4,
#         num_gpus=1,
#         mem="300g",
#         gpu_type="a100_80gb",
#         name="mnist_distillation",
#         out_file="mnist_distillation_out.txt",
#         err_file="mnist_distillation_err.txt",
#         mail_notification_on_start="krishnateja.killamsetty1@ibm.com",
#         mail_log_file_when_done="krishnateja.killamsetty1@ibm.com")


from subprocess import call

# call(["python", "main_ift.py", "--mode", "distill_basic", "--dataset", "MNIST", "--arch", "LeNet", "--init_image", "random", "--distill_lr", "0.02", "--lr", "1", "--batch_size", "5120", "--test_batch_size", "1024", "--epochs", "400", "--distill_epochs", "3", "--evaluate_epochs", "20", "--weight_decay", "5e-4", "--neumann_terms_cnt", "20", "--log_interval", "100"])
call(["python", "main_zen.py", "--mode", "distill_basic", "--dataset", "MNIST", "--zen_archs", "LeNet,VGG16,ResNet18,ConvNet", "--init_image", "random", "--distill_lr", "0.02", "--lr", "1", "--batch_size", "5120", "--test_batch_size", "1024", "--epochs", "400", "--distill_epochs", "5", "--evaluate_epochs", "20", "--weight_decay", "5e-4", "--neumann_terms_cnt", "20", "--log_interval", "100"])