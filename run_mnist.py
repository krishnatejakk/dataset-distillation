from cvar_pyutils.ccc import submit_job

submit_job(command_to_run="python main.py --mode distill_basic --dataset MNIST --arch LeNet",
        duration="12h",
        num_nodes=1,
        num_cores=4,
        num_gpus=1,
        mem="300g",
        gpu_type="a100_80gb",
        name="mnist_distillation",
        out_file="mnist_distillation_out.txt",
        err_file="mnist_distillation_err.txt",
        mail_notification_on_start="krishnateja.killamsetty1@ibm.com",
        mail_log_file_when_done="krishnateja.killamsetty1@ibm.com")