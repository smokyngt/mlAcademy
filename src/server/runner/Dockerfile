# Sandbox image for server-side exercises and notebook cells (NumPy + CPU PyTorch).
#   docker build -t ml-academy-runner src/server/runner
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# No entrypoint: the server passes `python -I /job/runner.py` with a read-only job mount.
WORKDIR /tmp
