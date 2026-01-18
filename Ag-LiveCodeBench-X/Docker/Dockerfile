##################################################
# Base image
FROM alpine:3.19

##################################################
# Install C compilation tools, GMP, and Python with verbose output
#
# - apk add -v    → verbose progress/info while installing
# - apk add -vv   → even more verbose
RUN apk add --no-cache -v \
    gcc \
    musl-dev \
    gmp-dev \
    python3

##################################################
# Create working directory
WORKDIR /workspace

##################################################
# Download uthash header
#
RUN mkdir -p /usr/local/include \
 && wget -O /usr/local/include/uthash.h \
    https://raw.githubusercontent.com/troydhanson/uthash/master/src/uthash.h

##################################################
# Copy the verifier script
COPY verify.py /usr/local/bin/verify.py

##################################################
# Make the verifier executable
RUN chmod +x /usr/local/bin/verify.py

##################################################
# Set entrypoint to the verifier
ENTRYPOINT ["python3", "/usr/local/bin/verify.py"]
