import wandb

api = wandb.Api()
artifact_versions = api.artifact_versions("model", "jbarry-team/rf-joint/run-hobhh3xe-events")

# Keep the most recent (assumed to be the first in the list)
# for artifact in list(artifact_versions)[1:]:
for artifact in list(artifact_versions):
    artifact.delete()
