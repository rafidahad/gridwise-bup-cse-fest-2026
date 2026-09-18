# GridWise fallback image (Guide S02 "Docker fallback image"; plan S12).
#
# Two deliberate choices:
#   * Python is pinned to 3.12-slim, the version named in plan S3.
#   * CBC is installed from Debian (coinor-cbc) *in addition to* the solver
#     binary bundled in the PuLP wheel. Plan S12 requires the working solver be
#     present in the image, and a distro package removes the risk of the
#     bundled binary failing against this base image's libc/libstdc++.
#
# No secret is ever baked in. Credentials are injected at run time through the
# environment-variable names documented in .env.example and README.md.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GRIDWISE_HOST=0.0.0.0 \
    GRIDWISE_PORT=8000

# coinor-cbc  -> the MILP/LP solver PuLP shells out to
# libgomp1     -> OpenMP runtime CBC links against
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        coinor-cbc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/gridwise

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY scripts ./scripts

# Build-time proof that the packaged solver actually runs, so a broken image
# fails here rather than on the first request a judge makes.
RUN python -c "import pulp; s = pulp.PULP_CBC_CMD(msg=False); assert s.available(), 'CBC unavailable'; print('CBC OK')"

# Run unprivileged.
RUN useradd --create-home --uid 10001 gridwise \
    && chown -R gridwise:gridwise /srv/gridwise
USER gridwise

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Bind 0.0.0.0 so the judge can reach the service from outside the container
# (Guide S02).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
