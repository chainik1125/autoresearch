"""Pytest fixtures: parametrize over LocalStorage and a moto-backed S3Storage so
every state-layer test exercises both backends identically."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from moto import mock_aws

from autoresearch.backends.storage import LocalStorage, S3Storage, StorageBackend


@pytest.fixture
def local_storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path / "store")


@pytest.fixture
def s3_storage() -> Iterator[S3Storage]:
    with mock_aws():
        storage = S3Storage(
            bucket="test-bucket",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        storage._client.create_bucket(Bucket="test-bucket")
        yield storage


@pytest.fixture(params=["local", "s3"])
def storage(
    request: pytest.FixtureRequest,
    local_storage: LocalStorage,
    s3_storage: S3Storage,
) -> StorageBackend:
    return local_storage if request.param == "local" else s3_storage
