import wandb

# Initialize the W&B run (if not already initialized)
# If you want to upload to an existing run, you'll need its ID
run = wandb.init(project='simexp-03', id='5iaubbci', resume='must')

# Upload the file
run.save('/cephfs/store/gr-mc2473/lc865/workspace/wandb-cache/wandb/latest-run/files/last.ckpt', 
         base_path='/cephfs/store/gr-mc2473/lc865/workspace/wandb-cache/wandb/latest-run/files/')

# Optional: Close the run when done
run.finish()