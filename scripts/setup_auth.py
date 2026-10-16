# Copyright (c) 2026 Juan Medina
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import json
import re
import subprocess
import sys
from pathlib import Path
from ytmusicapi import YTMusic
from ytmusicapi.helpers import initialize_headers, sapisid_from_cookie, get_authorization


def parse_curl_or_text(text: str) -> dict:
    headers = {}

    # Clean up Windows CMD escapes before parsing
    text = text.replace('^"', '"')
    
    # Extract cookies from curl -b or --cookie
    cookie_match = re.search(r"(?:-b|--cookie)\s+['\"]([^'\"]+)['\"]", text)
    if cookie_match:
        headers["cookie"] = cookie_match.group(1)

    # Extract headers from curl -H or --header
    for m in re.findall(r"(?:-H|--header)\s+['\"]([^'\"]+)['\"]", text):
        if ":" in m:
            k, v = m.split(":", 1)
            k = k.strip().lower()
            if k != "authorization":
                headers[k] = v.strip()

    # Extract plain headers
    if "cookie" not in headers:
        for line in text.splitlines():
            if ":" in line and not line.startswith("http"):
                k, v = line.split(":", 1)
                k = k.strip().lower()
                if k != "authorization":
                    headers[k] = v.strip()

    return headers


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    raw = ""
    curl_file = PROJECT_ROOT / "curl.txt"
    if curl_file.exists() and curl_file.stat().st_size > 0:
        print("Reading from curl.txt...")
        raw = curl_file.read_text(encoding="utf-8").strip()
    else:
        print("Reading from clipboard...")
        try:
            raw = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
                text=True,
                encoding="utf-8",
            ).strip()
        except Exception:
            raw = ""

    if not raw:
        print("Nothing found in clipboard or curl.txt!")
        print("In Edge: Right-click 'browse' -> Copy -> 'Copy as cURL (bash)'")
        sys.exit(1)

    headers = parse_curl_or_text(raw)

    if "cookie" not in headers:
        print("\nERROR: No cookie found in clipboard.")
        print("In Edge: Right-click 'browse' -> Copy -> 'Copy as cURL (bash)'")
        sys.exit(1)

    cookie_val = headers["cookie"]
    try:
        sapisid = sapisid_from_cookie(cookie_val)
        headers["authorization"] = get_authorization(f"{sapisid} https://music.youtube.com")
    except Exception:
        pass

    if "x-goog-authuser" not in headers:
        headers["x-goog-authuser"] = "0"

    ignore_keys = {"host", "content-length", "accept-encoding"}
    for k in list(headers.keys()):
        if k.startswith("sec") or k in ignore_keys:
            headers.pop(k, None)

    headers.update(initialize_headers())

    out_file = PROJECT_ROOT / "browser.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(headers, f, indent=4, ensure_ascii=True)

    print(f"\nSaved '{out_file}'. Testing connection...")
    try:
        yt = YTMusic(str(out_file))
        playlists = yt.get_library_playlists(limit=1)
        print("\n==========================================")
        print(" SUCCESS! Authentication verified!")
        print(" Unlimited sync mode is now ACTIVE.")
        print("==========================================")
    except Exception as e:
        print(f"\nWarning: Verification error: {e}")


if __name__ == "__main__":
    main()
