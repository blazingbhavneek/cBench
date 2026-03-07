# C Standard Library Reference

## Memory Management

### malloc
```c
#include <stdlib.h>
void *malloc(size_t size);
```
Allocates `size` bytes of uninitialized memory. Returns a pointer to the allocated memory, or NULL if the allocation fails.

**Example:**
```c
int *arr = malloc(10 * sizeof(int));
if (arr == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
}
// Use arr...
free(arr);  // Don't forget to free!
```

### free
```c
#include <stdlib.h>
void free(void *ptr);
```
Deallocates memory previously allocated by malloc, calloc, or realloc.

**Important:** 
- Never free the same pointer twice
- Never free a pointer that wasn't allocated
- Set pointer to NULL after freeing to avoid use-after-free

### realloc
```c
#include <stdlib.h>
void *realloc(void *ptr, size_t new_size);
```
Changes the size of the memory block pointed to by ptr.

**Example:**
```c
int *arr = malloc(10 * sizeof(int));
// Resize to hold 20 integers
arr = realloc(arr, 20 * sizeof(int));
```

## String Functions

### strlen
```c
#include <string.h>
size_t strlen(const char *str);
```
Returns the length of the string (excluding null terminator).

### strcpy
```c
#include <string.h>
char *strcpy(char *dest, const char *src);
```
Copies the string pointed to by src to dest. **Warning:** Does not check buffer bounds!

### strcmp
```c
#include <string.h>
int strcmp(const char *s1, const char *s2);
```
Compares two strings. Returns:
- 0 if strings are equal
- < 0 if s1 is less than s2
- > 0 if s1 is greater than s2

### memset
```c
#include <string.h>
void *memset(void *ptr, int value, size_t num);
```
Sets the first `num` bytes of the memory block pointed to by ptr to the specified value.

**Example:**
```c
int arr[10];
memset(arr, 0, sizeof(arr));  // Zero-initialize array
```

## Input/Output

### printf
```c
#include <stdio.h>
int printf(const char *format, ...);
```
Writes formatted output to stdout.

**Common format specifiers:**
- `%d` - integer
- `%lld` - long long
- `%f` - float/double
- `%s` - string
- `%p` - pointer
- `%c` - character

### scanf
```c
#include <stdio.h>
int scanf(const char *format, ...);
```
Reads formatted input from stdin.

**Example:**
```c
int n;
scanf("%d", &n);  // Note the & for address

char str[100];
scanf("%s", str);  // No & for arrays (already a pointer)
```

### fgets
```c
#include <stdio.h>
char *fgets(char *str, int size, FILE *stream);
```
Reads a line from the specified stream.

**Example:**
```c
char line[256];
fgets(line, sizeof(line), stdin);  // Read line from stdin
```

## GMP Library (Arbitrary Precision)

### Initialization
```c
#include <gmp.h>
mpz_t x;
mpz_init(x);              // Initialize to 0
mpz_init_set_ui(x, 42);   // Initialize and set to unsigned int
mpz_init_set_si(x, -42);  // Initialize and set to signed int
mpz_init_set_str(x, "12345678901234567890", 10);  // From string
```

### Arithmetic
```c
mpz_add(result, a, b);      // result = a + b
mpz_sub(result, a, b);      // result = a - b
mpz_mul(result, a, b);      // result = a * b
mpz_mod(result, a, b);      // result = a % b
mpz_pow_ui(result, base, exp);  // result = base^exp
```

### Cleanup
```c
mpz_clear(x);  // Free memory used by mpz_t
```

## UTHash (Hash Tables)

UTHash is a header-only hash table library. Include it with:
```c
#include "uthash.h"
```

### Basic Usage
```c
struct hash_entry {
    int key;            // Key field (required)
    int value;          // Value field
    UT_hash_handle hh;  // Makes this structure hashable (required)
};

struct hash_entry *hash = NULL;  // Initialize to NULL

// Add entry
struct hash_entry *entry = malloc(sizeof(struct hash_entry));
entry->key = 42;
entry->value = 100;
HASH_ADD_INT(hash, key, entry);

// Find entry
struct hash_entry *found;
HASH_FIND_INT(hash, &key, found);
if (found) {
    printf("Found: %d\n", found->value);
}

// Delete entry
HASH_DEL(hash, entry);
free(entry);

// Free entire hash
struct hash_entry *current, *tmp;
HASH_ITER(hh, hash, current, tmp) {
    HASH_DEL(hash, current);
    free(current);
}
```

## Common Patterns

### Reading Multiple Integers
```c
int n;
scanf("%d", &n);
int *arr = malloc(n * sizeof(int));
for (int i = 0; i < n; i++) {
    scanf("%d", &arr[i]);
}
// Process array...
free(arr);
```

### String Processing
```c
char str[1000];
scanf("%s", str);
int len = strlen(str);

// Convert to lowercase
for (int i = 0; i < len; i++) {
    str[i] = tolower(str[i]);
}
```

### Sorting with qsort
```c
int compare(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);  // Ascending order
}

int arr[] = {5, 2, 8, 1, 9};
int n = sizeof(arr) / sizeof(arr[0]);
qsort(arr, n, sizeof(int), compare);
```

### Two-Pointer Technique
```c
int left = 0, right = n - 1;
while (left < right) {
    // Process arr[left] and arr[right]
    if (condition) {
        left++;
    } else {
        right--;
    }
}
```
