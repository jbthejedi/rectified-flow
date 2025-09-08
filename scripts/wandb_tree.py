#!/usr/bin/env python3
# wandb_du.py
# Recursively list Weights & Biases project storage usage (runs + artifacts).

import os
import csv
import argparse
from collections import defaultdict
from pathlib import PurePosixPath

import wandb
from wandb.apis.public import Api
from rich.tree import Tree
from rich.console import Console

console = Console()

def human_bytes(n: int) -> str:
  if n is None:
    return "?"
  units = ["B", "KB", "MB", "GB", "TB"]
  i = 0
  f = float(n)
  while f >= 1024 and i < len(units) - 1:
    f /= 1024.0
    i += 1
  return f"{f:.2f} {units[i]}"

class SizeTree:
  def __init__(self, name="/", kind=""):
    self.name = name
    self.kind = kind  # e.g., "run", "artifact"
    self.size = 0
    self.children = {}

  def add(self, path: str, bytes_: int, kind: str):
    self.size += bytes_
    parts = [p for p in PurePosixPath(path).parts if p not in ("/", "")]
    if not parts:
      return
    head, tail = parts[0], parts[1:]
    if head not in self.children:
      self.children[head] = SizeTree(head, kind=kind)
    self.children[head].add("/".join(tail), bytes_, kind)

  def render(self, max_depth=3, prefix=""):
    # returns a rich Tree
    label = f"{self.name}  [{self.kind}]  {human_bytes(self.size)}"
    tree = Tree(label)
    if max_depth <= 0:
      return tree
    for name, child in sorted(self.children.items(), key=lambda kv: kv[1].size, reverse=True):
      subtree = child.render(max_depth - 1)
      tree.add(subtree)
    return tree

  def to_rows(self, base_path=""):
    rows = []
    path = f"{base_path}/{self.name}".strip("/")
    rows.append((path, self.kind, self.size, human_bytes(self.size)))
    for child in self.children.values():
      rows.extend(child.to_rows(path))
    return rows

def collect_runs(api: Api, entity: str, project: str, show_files: bool, size_tree: SizeTree):
  path = f"{entity}/{project}"
  console.print(f"[bold]Scanning runs[/bold] for {path} ...")
  runs = api.runs(path, per_page=2000)
  for run in runs:
    run_prefix = f"runs/{run.id}"
    total_run_bytes = 0
    try:
      files = list(run.files())  # generator → list
    except Exception as e:
      console.print(f"[yellow]Warning:[/yellow] could not list files for run {run.id}: {e}")
      continue
    for f in files:
      # f.name is a POSIX-like path (e.g., 'media/images/xx.png')
      try:
        size = int(getattr(f, "size", 0) or 0)
      except Exception:
        size = 0
      total_run_bytes += size
      # group by directory prefixes
      parts = PurePosixPath(f.name).parts
      if show_files:
        rel_path = "/".join((run_prefix, "files", f.name))
        size_tree.add(rel_path, size, kind="run-file")
      else:
        # bucket by top-level dir (media/, tables/, logs/, etc.)
        top = parts[0] if parts else ""
        rel_path = "/".join((run_prefix, top if top else "files"))
        size_tree.add(rel_path, size, kind="run-dir")
    # Also add a node for the run total
    size_tree.add(f"{run_prefix}", total_run_bytes, kind="run-total")

def _artifact_size_safe(artifact) -> int:
  # Prefer artifact.size; if missing, sum manifest entries.
  try:
    if getattr(artifact, "size", None):
      return int(artifact.size)
  except Exception:
    pass
  # Fallback: manifest entries
  total = 0
  try:
    manifest = artifact.manifest
    if manifest and getattr(manifest, "entries", None):
      for ent in manifest.entries.values():
        try:
          total += int(getattr(ent, "size", 0) or 0)
        except Exception:
          pass
  except Exception:
    pass
  return total

def collect_artifacts(api: Api, entity: str, project: str, size_tree: SizeTree):
  console.print(f"[bold]Scanning artifacts[/bold] for {entity}/{project} ...")
  # List artifact types → collections → versions
  try:
    types = api.artifacts_types(f"{entity}/{project}")
  except Exception as e:
    console.print(f"[yellow]Warning:[/yellow] could not list artifact types: {e}")
    return
  for t in types:
    tname = getattr(t, "name", "type")
    try:
      collections = list(t.collections())
    except Exception as e:
      console.print(f"[yellow]Warning:[/yellow] could not list collections for type {tname}: {e}")
      continue
    for coll in collections:
      cname = getattr(coll, "name", "collection")
      # Iterate versions (pages)
      try:
        versions = list(coll.versions(per_page=200))
      except TypeError:
        versions = list(coll.versions())
      except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] could not list versions for {tname}/{cname}: {e}")
        continue
      for art in versions:
        aname = getattr(art, "name", "artifact")
        aliases = []
        try:
          aliases = [a.name for a in art.aliases or []]
        except Exception:
          pass
        alias_str = f"@{','.join(aliases)}" if aliases else ""
        bytes_ = _artifact_size_safe(art)
        # Tree paths:
        base = f"artifacts/{tname}/{cname}"
        size_tree.add(f"{base}", bytes_, kind="artifact-collection")
        size_tree.add(f"{base}/{aname}{alias_str}", bytes_, kind="artifact-version")

def main():
  parser = argparse.ArgumentParser(description="W&B project disk usage (runs + artifacts).")
  parser.add_argument("--entity", required=True, help="W&B entity (team/user)")
  parser.add_argument("--project", required=True, help="W&B project")
  parser.add_argument("--max-depth", type=int, default=3, help="Depth for tree view")
  parser.add_argument("--show-files", action="store_true", help="Include individual run files")
  parser.add_argument("--out-csv", default="sizes.csv", help="CSV path to write")
  args = parser.parse_args()

  # Ensure API key exists
  if not os.environ.get("WANDB_API_KEY"):
    console.print("[red]WANDB_API_KEY is not set in environment.[/red]")
    return

  api = wandb.Api()
  root = SizeTree(name=f"{args.entity}/{args.project}", kind="project")

  # Collect data
  collect_runs(api, args.entity, args.project, args.show_files, root)
  collect_artifacts(api, args.entity, args.project, root)

  # Render tree
  console.print("\n[bold underline]Storage Tree[/bold underline]")
  tree = root.render(max_depth=args.max_depth)
  console.print(tree)

  # Write CSV
  rows = root.to_rows()
  with open(args.out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "kind", "bytes", "human_size"])
    # de-duplicate identical nodes (same path/kind) by keeping max bytes
    agg = {}
    for p, k, b, hb in rows:
      key = (p, k)
      agg[key] = max(agg.get(key, 0), int(b))
    for (p, k), b in sorted(agg.items(), key=lambda kv: kv[1], reverse=True):
      writer.writerow([p, k, b, human_bytes(b)])

  console.print(f"\n[green]Wrote CSV:[/green] {args.out_csv}")
  console.print("[dim]Tip: sort by bytes desc to spot the hogs. Delete big runs or artifact versions/aliases in the W&B UI (or via API).[/dim]")

if __name__ == "__main__":
  main()
