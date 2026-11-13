import hashlib
from datetime import datetime, timezone
from contextvars import ContextVar
_ARCHIVE = ContextVar("demo_archive", default=None)
def bind_archive(archive):
    _ARCHIVE.set(archive)
def archive_csv(csv_content, schema_name, table_name):
    archive = _ARCHIVE.get()
    if archive is None: raise RuntimeError("Demo archive not initialized")
    data = csv_content.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    prefix = f"csv-uploads/{schema_name}/{table_name}/"
    for key, record in archive.items():
        if key.startswith(prefix) and record['sha256'] == digest:
            return {'upload_status':'duplicate','s3_key':key,'status':'Identical CSV already exists in simulated S3. Upload skipped.'}
    now = datetime.now(timezone.utc)
    key = f"{prefix}{now:%Y-%m-%d}/{table_name}_{now:%Y-%m-%d_%H-%M-%S-%f}_{digest[:12]}.csv"
    archive[key] = {'sha256':digest,'content':data}
    return {'upload_status':'success','s3_key':key,'status':'CSV archived in simulated S3 (current session only).'}
