import os

env_vars = [
    "SLURM_JOB_ID",
    "SLURM_JOBID",
    "SLURM_SUBMIT_DIR",
    "SLURM_SUBMIT_HOST",
    "SLURM_JOB_NODELIST",
    "SLURM_NODELIST",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "SLURM_JOB_CPUS_PER_NODE",
    "SLURM_CPUS_PER_GPU",
    "SLURM_MEM_PER_CPU",
    "SLURM_MEM_PER_GPU",
    "SLURM_MEM_PER_NODE",
    "SLURM_GPUS",
    "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_NTASKS_PER_SOCKET",
    "SLURM_NTASKS_PER_CORE",
    "SLURM_NTASKS_PER_GPU",
    "SLURM_NPROCS",
    "SLURM_NNODES",
    "SLURM_TASKS_PER_NODE",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_ARRAY_TASK_COUNT",
    "SLURM_ARRAY_TASK_MAX",
    "SLURM_ARRAY_TASK_MIN"
]

def check_env_vars():
    for var in env_vars:
        value = os.environ.get(var)
        if value is not None:
            print(f"{var}: {value}")
        else:
            print(f"{var}: Does not exist")

if __name__ == "__main__":
    check_env_vars()