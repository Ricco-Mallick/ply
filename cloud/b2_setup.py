"""Create (or update) the private Backblaze B2 bucket the Ply API stores captures and results in.

Reads B2_KEY_ID / B2_APP_KEY (an application key with access to all buckets, read + write) from cloud/.b2.
Prints shell assignments for deploy.sh: S3_ENDPOINT, S3_REGION, S3_BUCKET.
CORS lets the capture app PUT parts straight to the bucket through signed URLs and read the ETag back.
Standard library only.
"""

import base64
import json
import secrets
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
CORS = [{
    "corsRuleName": "ply-app",
    "allowedOrigins": ["*"],  # every request is a signed URL scoped to one object; origin adds nothing
    "allowedOperations": ["s3_put", "s3_get", "s3_head"],
    "allowedHeaders": ["*"],
    "exposeHeaders": ["ETag"],
    "maxAgeSeconds": 3600,
}]


def env(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def call(url: str, token: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Authorization": token, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"B2 {url.rsplit('/', 1)[-1]} failed: {e.code} {e.read()[:300]!r}")


def main() -> None:
    cfg_path = HERE / ".b2"
    if not cfg_path.exists():
        raise SystemExit("cloud/.b2 is missing: put B2_KEY_ID=... and B2_APP_KEY=... in it")
    cfg = env(cfg_path)
    basic = base64.b64encode(f"{cfg['B2_KEY_ID']}:{cfg['B2_APP_KEY']}".encode()).decode()
    try:
        with urllib.request.urlopen(urllib.request.Request("https://api.backblazeb2.com/b2api/v3/b2_authorize_account", headers={"Authorization": f"Basic {basic}"}), timeout=60) as r:
            auth = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"B2 rejected the key: {e.code} {e.read()[:200]!r}")
    api = auth["apiInfo"]["storageApi"]
    base, token, account = api["apiUrl"] + "/b2api/v3", auth["authorizationToken"], auth["accountId"]
    s3 = api["s3ApiUrl"]  # e.g. https://s3.us-west-004.backblazeb2.com
    region = s3.split("//", 1)[1].split(".")[1]

    name = cfg.get("B2_BUCKET")
    existing = {b["bucketName"]: b for b in call(base + "/b2_list_buckets", token, {"accountId": account})["buckets"]}
    if name and name in existing:
        call(base + "/b2_update_bucket", token, {"accountId": account, "bucketId": existing[name]["bucketId"], "bucketType": "allPrivate", "corsRules": CORS})
    else:
        name = name or f"ply-data-{secrets.token_hex(4)}"  # bucket names are global across B2
        call(base + "/b2_create_bucket", token, {"accountId": account, "bucketName": name, "bucketType": "allPrivate", "corsRules": CORS,
                                                  "lifecycleRules": [{"fileNamePrefix": "captures/", "daysFromUploadingToHidden": 30, "daysFromHidingToDeleting": 1}]})
        with cfg_path.open("a") as f:
            f.write(f"B2_BUCKET={name}\n")
    print(f"S3_ENDPOINT={s3}\nS3_REGION={region}\nS3_BUCKET={name}")


if __name__ == "__main__":
    sys.exit(main())
