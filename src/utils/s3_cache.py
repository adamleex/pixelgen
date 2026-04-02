"""Multi-threaded S3 download helpers for model weights.

Supports two backends:
  1. boto3 (preferred) — uses TransferConfig for parallel multipart downloads
  2. pyarrow.fs — manual chunked parallel download via ThreadPoolExecutor

Usage:
    from src.utils.s3_cache import resolve_weight_path

    # In model __init__:
    local_path = resolve_weight_path("s3://bucket/path/to/weights.pth")
    state_dict = torch.load(local_path, map_location="cpu")
"""

import hashlib
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

_HAS_AWS_CLI = shutil.which("aws") is not None

CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB per chunk
MAX_WORKERS = 8


def _local_cache_path(s3_uri: str) -> str:
    """Map an s3:// URI to a deterministic local cache path under ~/.cache/s3_weights/."""
    s3_path = s3_uri.replace("s3://", "", 1)
    bucket, _, key = s3_path.partition("/")
    filename = os.path.basename(key)
    uri_hash = hashlib.md5(s3_uri.encode()).hexdigest()[:10]
    cache_dir = os.path.expanduser(f"~/.cache/s3_weights/{bucket}/{uri_hash}")
    return os.path.join(cache_dir, filename)


def _download_aws_cli(s3_uri: str, local_path: str) -> None:
    print(f"[s3_cache] aws s3 cp {s3_uri} -> {local_path}", flush=True)
    subprocess.run(["aws", "s3", "cp", s3_uri, local_path], check=True)


def _download_boto3(s3_uri: str, local_path: str) -> None:
    import boto3
    from boto3.s3.transfer import TransferConfig

    s3_path = s3_uri.replace("s3://", "", 1)
    bucket, _, key = s3_path.partition("/")

    config = TransferConfig(
        multipart_threshold=CHUNK_SIZE,
        multipart_chunksize=CHUNK_SIZE,
        max_concurrency=MAX_WORKERS,
        use_threads=True,
    )

    s3 = boto3.client("s3")
    size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    print(f"[s3_cache] boto3 download {s3_uri} ({size / 1024 / 1024:.1f} MB, {MAX_WORKERS} threads)", flush=True)
    s3.download_file(bucket, key, local_path, Config=config)


def _download_pyarrow(s3_uri: str, local_path: str) -> None:
    """Multi-threaded chunked download using pyarrow S3 filesystem."""
    import pyarrow.fs

    s3_path = s3_uri.replace("s3://", "", 1)
    fs = pyarrow.fs.S3FileSystem(region="us-east-1")
    fi = fs.get_file_info(s3_path)
    total_size = fi.size
    print(
        f"[s3_cache] pyarrow download {s3_uri} ({total_size / 1024 / 1024:.1f} MB, {MAX_WORKERS} threads)",
        flush=True,
    )

    num_chunks = max(1, (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE)
    if num_chunks == 1:
        with fs.open_input_stream(s3_path) as src, open(local_path, "wb") as dst:
            dst.write(src.read())
        return

    tmp_dir = local_path + ".parts"
    os.makedirs(tmp_dir, exist_ok=True)

    def _download_chunk(idx: int) -> str:
        offset = idx * CHUNK_SIZE
        length = min(CHUNK_SIZE, total_size - offset)
        part_path = os.path.join(tmp_dir, f"part_{idx:04d}")
        chunk_fs = pyarrow.fs.S3FileSystem(region="us-east-1")
        with chunk_fs.open_input_stream(s3_path) as stream:
            stream.seek(offset)
            data = stream.read(length)
        with open(part_path, "wb") as f:
            f.write(data)
        return part_path

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_download_chunk, i): i for i in range(num_chunks)}
            done_count = 0
            for future in as_completed(futures):
                future.result()
                done_count += 1
                if done_count % max(1, num_chunks // 5) == 0 or done_count == num_chunks:
                    print(f"[s3_cache]   progress: {done_count}/{num_chunks} chunks", flush=True)

        with open(local_path, "wb") as out:
            for i in range(num_chunks):
                part_path = os.path.join(tmp_dir, f"part_{i:04d}")
                with open(part_path, "rb") as part:
                    out.write(part.read())
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def download_s3(s3_uri: str, local_path: str) -> None:
    """Download a file from S3 to local_path using the fastest available method."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp_path = local_path + ".downloading"

    try:
        if _HAS_AWS_CLI:
            _download_aws_cli(s3_uri, tmp_path)
        else:
            try:
                _download_boto3(s3_uri, tmp_path)
            except Exception as e:
                print(f"[s3_cache] boto3 failed ({e}), falling back to pyarrow", flush=True)
                _download_pyarrow(s3_uri, tmp_path)

        os.replace(tmp_path, local_path)
        print(f"[s3_cache] done: {local_path} ({os.path.getsize(local_path) / 1024 / 1024:.1f} MB)", flush=True)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def resolve_weight_path(path: str) -> str:
    """Resolve a weight path: download from S3 if needed, expand ~ otherwise.

    If path starts with s3://, downloads to local cache and returns the local path.
    Skips download if the cached file already exists.
    """
    if not path:
        return path

    if path.startswith("s3://"):
        local_path = _local_cache_path(path)
        if not os.path.isfile(local_path):
            download_s3(path, local_path)
        else:
            print(f"[s3_cache] using cached: {local_path}", flush=True)
        return local_path

    return os.path.expanduser(path)
