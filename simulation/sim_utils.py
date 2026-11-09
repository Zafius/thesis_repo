"""
sim_utils.py -- small run-lifecycle helpers shared by the simulation entry
points: out-of-band progress notifications and post-run archiving.

Replaces the former one-function modules Notifications.py and
compressing_files.py. Plotting deliberately stayed in csv_to_plot.py -- it is a
cohesive matplotlib module in its own right, not a utility.

Neither helper here is allowed to break a run: a simulation batch can take
hours, so a missing webhook, a dead network or a locked file must degrade to a
printed warning rather than lose the work.
"""
import os
import shutil

import requests

# Webhook is read from the environment, never hardcoded -- see .env (gitignored).
# Unset simply disables notifications, which is the right default for anyone
# who clones the repo and does not want to post into someone else's channel.
WEBHOOK_ENV_VAR = "DISCORD_WEBHOOK_URL"

# Load ../.env so the variable can live in the gitignored .env alongside the
# other secrets, not just in the shell. Optional dependency: without it the
# variable simply has to be exported by the caller.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

_warned_missing_webhook = False


def send_discord_alert(message):
    """Post a progress/failure message to the Discord channel in
    $DISCORD_WEBHOOK_URL. No-op (with a single warning per process) when the
    variable is unset."""
    global _warned_missing_webhook

    webhook_url = os.environ.get(WEBHOOK_ENV_VAR)
    if not webhook_url:
        if not _warned_missing_webhook:
            print(f"[notify] ${WEBHOOK_ENV_VAR} not set -- Discord notifications disabled.")
            _warned_missing_webhook = True
        return

    data = {"content": message, "username": "Python Script Bot"}
    try:
        result = requests.post(webhook_url, json=data, timeout=10)
        if result.status_code == 204:
            print("Notification sent!")
        else:
            print(f"Failed to send: {result.status_code}")
    except requests.RequestException as e:
        print(f"Discord notification skipped (no connection?): {e}")


def compress_and_delete(folder_path, output_name):
    """Zip folder_path to <output_name>.zip, then delete the original.

    The delete only happens if make_archive returned without raising, so a
    failure mid-compression leaves the source folder untouched rather than
    destroying results that were never successfully archived.
    """
    try:
        shutil.make_archive(output_name, "zip", folder_path)
        shutil.rmtree(folder_path)
    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' does not exist.")
    except Exception as e:
        # Permissions errors, full disk, file locked by a viewer, ...
        print(f"An error occurred: {e}. Aborting deletion to keep your files safe.")


if __name__ == "__main__":
    send_discord_alert("✅ sim_utils.py notification test.")
