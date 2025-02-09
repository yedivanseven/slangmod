# Start with a build stage
FROM python:3.12-slim as build

# Suppress pipenv complaints
ARG PIPENV_VERBOSITY=-1
# This should be supplied as a build argument!
ARG SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1

# Copy the repository contents
WORKDIR /usr/local/src
COPY . .

# Install dependencies and package into a virtual environment
RUN python -m venv /usr/local/venv
RUN . /usr/local/venv/bin/activate && \
    pip install --upgrade pip pipenv && \
    pipenv sync --categories="packages cuda" && \
    pip install .


# Continue with a clean install stage
FROM python:3.12-slim

# Copy the virtual environment and put it on the path
COPY --from=build /usr/local/venv /usr/local/venv
ENV PATH="/usr/local/venv/bin:$PATH"

# Documents we still nedd to clean must be mounted to /raw
ENV files="{'raw': '/raw'}"

# Create a user to run as
USER 1937:2846

# A working directory with the slangmod.toml in it must be mounted to /workdir
WORKDIR /workdir

# The entrypoint is the package itself
ENTRYPOINT ["/usr/local/venv/bin/slangmod"]
CMD ["--help"]
