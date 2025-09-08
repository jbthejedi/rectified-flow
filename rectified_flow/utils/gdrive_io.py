# gdrive_io.py
import os, torch
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request


# Minimal scope: lets you create/update files your app owns
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def auth_drive(client_secret_path="client_secret.json", token_path="token.json"):
  """
    First run prints a URL + code (device flow) in the console (works on SSH/Runpod).
    Token is cached to token.json for subsequent runs.
    """
  creds = None
  if os.path.exists(token_path):
    creds = Credentials.from_authorized_user_file(token_path, SCOPES)
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
      # Console/device flow—copy/paste the code once; then it's cached
      creds = flow.run_console()
    with open(token_path, "w") as f:
      f.write(creds.to_json())
  return build("drive", "v3", credentials=creds)


def _find_child_folder(service, parent_id, name):
  q = f"'{parent_id}' in parents and name = '{name}' and mimeType = 'application/vnd.google-apps.folder' and trashed=false"
  r = service.files().list(q=q, fields="files(id)", pageSize=1).execute()
  files = r.get("files", [])
  return files[0]["id"] if files else None


def ensure_path(service, drive_path, root_id="root"):
  """
    Creates nested folders under 'My Drive' to match drive_path ('ckpts/run_001', etc.).
    Returns the leaf folder id.
    """
  parent = root_id
  for part in [p for p in drive_path.split("/") if p]:
    found = _find_child_folder(service, parent, part)
    if not found:
      meta = {"name": part, "mimeType": "application/vnd.google-apps.folder", "parents": [parent]}
      found = service.files().create(body=meta, fields="id").execute()["id"]
    parent = found
  return parent


def save_and_upload_model(service, model, config, drive_path, filename="best-model.pth"):
  """
    Saves model locally then uploads to Google Drive at drive_path/filename.
    If a file with the same name exists in that folder, it updates in place.
    """
  to_save = model._orig_mod if getattr(config, "compile", False) and hasattr(model, "_orig_mod") else model
  torch.save(to_save.state_dict(), filename)

  folder_id = ensure_path(service, drive_path)  # e.g., "rf_ckpts/exp_42"
  q = f"'{folder_id}' in parents and name = '{filename}' and trashed=false"
  r = service.files().list(q=q, fields="files(id)", pageSize=1).execute()
  media = MediaFileUpload(filename, mimetype="application/octet-stream", resumable=False)

  if r.get("files"):
    file_id = r["files"][0]["id"]
    service.files().update(fileId=file_id, media_body=media).execute()
    return file_id
  else:
    meta = {"name": filename, "parents": [folder_id]}
    file = service.files().create(body=meta, media_body=media, fields="id").execute()
    return file["id"]


def write_smoke_test():
  service = auth_drive(client_secret_path="client_secret.json", token_path="token.json")
  # 1) create nested folders under "My Drive"
  folder_id = ensure_path(service, "rf_ckpts/test_upload")

  # 2) make a tiny file and upload it
  fname = "hello.txt"
  with open(fname, "w") as f:
      f.write("it works ✅\n")

  media = MediaFileUpload(fname, mimetype="text/plain", resumable=False)

  # upsert by name in that folder
  q = f"'{folder_id}' in parents and name='{fname}' and trashed=false"
  r = service.files().list(q=q, fields="files(id)", pageSize=1).execute()
  if r.get("files"):
      file_id = r["files"][0]["id"]
      service.files().update(fileId=file_id, media_body=media).execute()
  else:
      meta = {"name": fname, "parents": [folder_id]}
      file_id = service.files().create(body=meta, media_body=media, fields="id").execute()["id"]

  print("Uploaded. file_id:", file_id)


def authenticate():
  SCOPES = ["https://www.googleapis.com/auth/drive.file"]
  path = "/Users/justinbarry/projects/google_apps/rf_desktop_app/client_secret.json"

  flow = InstalledAppFlow.from_client_secrets_file(path, SCOPES)
  # Starts a tiny HTTP server on localhost and opens a browser to complete auth
  creds = flow.run_local_server(port=0, prompt="consent")  # returns Credentials
  service = build("drive", "v3", credentials=creds)
  with open("token.json", "w") as f:
      f.write(creds.to_json())
  print("Auth OK — token.json written.")


def main():
  # write_smoke_test()
  authenticate()

if __name__ == '__main__':
  main()