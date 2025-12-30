# Dockerfile for Agnostics C Verifier Container with GLib support
# This container can compile and execute C code following the Agnostics protocol

FROM alpine:3.19

# Install necessary packages:
# - gcc and musl-dev for C compilation
# - python3 for running the verifier script
# - glib and glib-dev for standard data structures
# - pkgconfig for getting GLib compilation flags
RUN apk add --no-cache \
    gcc \
    musl-dev \
    python3 \
    glib \
    glib-dev \
    pkgconfig

# Create working directory
WORKDIR /workspace

# Copy the verifier script
COPY verify.py /usr/local/bin/verify.py

# Make the verifier executable
RUN chmod +x /usr/local/bin/verify.py

# Set the entrypoint to the verifier script
ENTRYPOINT ["python3", "/usr/local/bin/verify.py"]
