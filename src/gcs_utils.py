from pathlib import Path

from google.cloud import storage


class GcsClient:
    def __init__(self, bucket_name: str):
        self.client = storage.Client()
        self.bucket_name = bucket_name
        self.bucket = self.client.bucket(bucket_name)

        if not self.bucket.exists():
            raise ValueError(f"GCS bucket does not exist or is not accessible: {bucket_name}")

    def upload_file(
        self,
        local_path: Path,
        destination_blob_name: str,
        skip_if_exists: bool = True,
    ) -> str:
        blob = self.bucket.blob(destination_blob_name)

        if skip_if_exists and blob.exists():
            return f"gs://{self.bucket_name}/{destination_blob_name}"

        blob.upload_from_filename(
            str(local_path),
            checksum="auto",
        )

        blob.reload()

        if blob.size is None or blob.size <= 0:
            raise RuntimeError(f"Uploaded blob has invalid size: {destination_blob_name}")

        return f"gs://{self.bucket_name}/{destination_blob_name}"

    def blob_exists(self, blob_name: str) -> bool:
        blob = self.bucket.blob(blob_name)
        return blob.exists()

    def get_blob_size(self, blob_name: str) -> int | None:
        blob = self.bucket.blob(blob_name)

        if not blob.exists():
            return None

        blob.reload()
        return blob.size