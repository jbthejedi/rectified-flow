import wandb

api = wandb.Api()
id = "d37tpdbe"

# server-unet-pp-tc-pixel-img32-flickr30k-cfg
# run_path = "jbarry-team/rf-joint/ejd38eou"
run_path = 'jbarry-team/rf-joint/0uylmgg7' 

run = api.run(run_path)
artifacts = run.logged_artifacts()
for art in artifacts:
    if "best" not in art.aliases:
      art.delete()
    else:
      print(art.name, art.type, art.version)