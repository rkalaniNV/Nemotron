# OOTB Dockerfile for Nemotron Customizer airgap packaging.
# Scope: workflows represented by src/nemotron/steps/.

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG AIRGAP_BUNDLE=deploy/nemotron-customizer/airgap/airgap-bundle
ARG PYTHON_BIN=python
ARG UV_VERSION=0.11.1

WORKDIR /workspace/Nemotron
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY ${AIRGAP_BUNDLE}/runtime/wheels/ /opt/nemotron-airgap/wheels/
COPY ${AIRGAP_BUNDLE}/runtime/requirements-airgap.txt /opt/nemotron-airgap/requirements-airgap.txt
COPY ${AIRGAP_BUNDLE}/runtime/offline.env /opt/nemotron-airgap/offline.env

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV PIP_NO_INDEX=1
ENV PIP_FIND_LINKS=/opt/nemotron-airgap/wheels
ENV UV_FIND_LINKS=/opt/nemotron-airgap/wheels
ENV UV_NO_INDEX=1
ENV UV_OFFLINE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_SYNC=1
ENV UV_PYTHON_DOWNLOADS=never
ENV WANDB_MODE=offline
ENV HF_HOME=/opt/nemotron-airgap/assets/hf-cache
ENV HF_HUB_CACHE=/opt/nemotron-airgap/assets/hf-cache/hub
ENV NEMOTRON_AIRGAP_ASSETS=/opt/nemotron-airgap/assets
ENV NEMOTRON_AIRGAP_REPOS=/opt/nemotron-airgap/assets/repos
ENV PYTHONPATH=/workspace/Nemotron/src:$PYTHONPATH
ENV PATH=/workspace/Nemotron/.venv/bin:$PATH

# Expect wheels to be prepared by:
#   nemotron step airgap fetch <lock> --include-wheels
# Only airgap-bundle/runtime is baked into this image. Large models, datasets,
# and git repos stay under airgap-bundle/assets and should be mounted at
# /opt/nemotron-airgap/assets when the step runs.
# BASE_IMAGE must provide ${PYTHON_BIN} with pip or ensurepip. If it does not,
# create a tiny derivative base image that enables pip first; all package
# installs below still use only /opt/nemotron-airgap/wheels.
RUN export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" && \
    (${PYTHON_BIN} -m pip --version >/dev/null 2>&1 || ${PYTHON_BIN} -m ensurepip --upgrade) && \
    ${PYTHON_BIN} -m pip install --no-index --find-links=/opt/nemotron-airgap/wheels "uv==${UV_VERSION}" && \
    uv venv --clear --python "$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')" .venv && \
    uv pip install --python .venv/bin/python --no-index \
        --find-links /opt/nemotron-airgap/wheels \
        -r /opt/nemotron-airgap/requirements-airgap.txt && \
    uv pip install --python .venv/bin/python --no-index \
        --find-links /opt/nemotron-airgap/wheels \
        --no-deps .
