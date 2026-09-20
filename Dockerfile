# Single multi-stage image for both training and serving.
#
# One `base` stage installs the dependencies and copies the source; the two
# runtime stages add only their entry point. The point is correctness: train
# and serve run byte-identical code and dependencies, which is the train/serve
# skew tests/unit/test_train_serve_feature_parity exists to catch.
#
# Storage sharing is a separate question and depends on the builder. BuildKit
# (CI, and Docker's default where buildx is present) dedupes the base layers;
# the classic builder re-runs the base stage per target, so building serve and
# train there costs ~1.7 GB each. Measured on a classic-builder daemon:
# serve alone 1.95 GB, serve + train 3.62 GB.
#
# No apt layer. Every pinned dependency ships a cp314 manylinux wheel
# (verified: catboost, numpy, pandas and the rest all install prebuilt), so
# build-essential bought nothing but size and a dependency on a Debian mirror.
# The health check uses the interpreter that is already here instead of curl.
#
# Build with `--network=host` where the default bridge has no DNS
# (docker-compose.yml sets it); the pip layer is the only step needing a
# network at all.
FROM python:3.14-slim AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    TZ=UTC \
    BL_REQUIRE_NAME_INDEX=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
# The compact name index. Without it the vendored scripts fall back to the
# real names_dataset: ~1.9 GB RSS and ~6 s of startup per process. With
# BL_REQUIRE_NAME_INDEX=1 above, a missing index fails loudly instead.
COPY assets/ assets/
COPY pyproject.toml .

# Baked at build time because .git is not in the build context, so the
# repo-reading fallback in train_pipeline._git_sha() cannot work inside an
# image. Without this, every containerised run logs git_sha="unknown" and the
# provenance param is worthless exactly where it matters most.
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}


FROM base AS serve
EXPOSE 8000
# /ready, not /health: /health only proves the process answers HTTP, while
# /ready also requires the startup canary to have produced a real prediction —
# the only check that catches an evicted hosted TabPFN fit.
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready', timeout=4).status == 200 else 1)"
CMD ["python", "scripts/serve.py"]


FROM base AS train
ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--mode", "production"]
