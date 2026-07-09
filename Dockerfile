FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Prophet 1.1.5 ships a precompiled model binary, but its bundled CmdStan omits the
# top-level `makefile` that cmdstanpy>=1.3 validates for on backend load. No runtime
# compilation is needed, so create the stub file to satisfy the check.
RUN python -c "import glob, os, prophet; d = glob.glob(os.path.join(os.path.dirname(prophet.__file__), 'stan_model', 'cmdstan-*'))[0]; open(os.path.join(d, 'makefile'), 'a').close()"

COPY . .

ENV PYTHONPATH=/app
