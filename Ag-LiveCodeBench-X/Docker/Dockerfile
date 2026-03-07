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
# Verify libraries are installed and usable by compiling and running a test C program
# This ensures GMP, math library, and uthash are properly configured
RUN cat > /tmp/verify_libs.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <gmp.h>
#include "uthash.h"

struct hash_entry {
    int key;
    int value;
    UT_hash_handle hh;
};

int main() {
    // Test GMP
    mpz_t a, b, result;
    mpz_init(a);
    mpz_init(b);
    mpz_init(result);
    mpz_set_ui(a, 12345);
    mpz_set_ui(b, 67890);
    mpz_mul(result, a, b);

    // Test math library
    double sqrt_val = sqrt(16.0);

    // Test uthash
    struct hash_entry *hash = NULL;
    struct hash_entry *entry = malloc(sizeof(struct hash_entry));
    entry->key = 1;
    entry->value = 100;
    HASH_ADD_INT(hash, key, entry);

    struct hash_entry *found;
    HASH_FIND_INT(hash, &entry->key, found);

    // Verify results
    if (mpz_cmp_ui(result, 838102050) == 0 &&
        sqrt_val == 4.0 &&
        found != NULL && found->value == 100) {
        printf("All libraries verified successfully!\n");
        printf("GMP: 12345 * 67890 = ");
        gmp_printf("%Zd\n", result);
        printf("Math: sqrt(16) = %.1f\n", sqrt_val);
        printf("Uthash: found entry with value = %d\n", found->value);
    } else {
        fprintf(stderr, "Library verification failed!\n");
        return 1;
    }

    // Cleanup
    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(result);
    HASH_DEL(hash, entry);
    free(entry);

    return 0;
}
EOF
RUN gcc -std=c11 -O2 -o /tmp/verify_libs /tmp/verify_libs.c -lm -lgmp && \
    /tmp/verify_libs && \
    echo "=== Library verification completed successfully ===" && \
    rm /tmp/verify_libs.c /tmp/verify_libs

##################################################
# Set entrypoint to the verifier
ENTRYPOINT ["python3", "/usr/local/bin/verify.py"]
