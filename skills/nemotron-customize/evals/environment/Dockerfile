# ACES/Harbor uses custom Dockerfiles in rebase mode for this eval. Keep this
# file as a small dependency layer over the managed skill-eval base image.

RUN python -m pip install --no-cache-dir \
    "datasets>=2.14" \
    "huggingface_hub>=0.20" \
    "hf_transfer>=0.1" \
    "pytest>=8" \
    "transformers>=4.40" \
    "accelerate>=0.30" \
    "sentencepiece>=0.2" \
    "sacrebleu>=2.6,<3" \
    "ray[default,data]>=2.54" \
    "sentence-transformers>=5,<6" \
    "nemo-curator @ git+https://github.com/NVIDIA-NeMo/Curator.git@main"
