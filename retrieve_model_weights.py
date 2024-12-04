import wandb

with open("src/wandb_apikey.txt", 'r') as f:
    wandb_api_key = f.read().strip()

wandb.login(key=wandb_api_key)
run = wandb.init(project='SePROFiT-Net', entity='cnmd-phb-postech')

artifact = run.use_artifact('pbe_u_z2ugi3bh:latest', type='model')
artifact_dir = artifact.download()
print(f"Artifact downloaded to: {artifact_dir}")
run.finish()
