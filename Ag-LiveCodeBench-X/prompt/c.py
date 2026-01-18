C_CRITICAL_CODING_REQUIREMENTS = """
CRITICAL REQUIREMENTS:

    1. INCLUDES - You MUST include ALL necessary headers:
       Standard C headers:
       - #include <stdio.h>      // for printf, scanf, FILE operations
       - #include <stdlib.h>     // for malloc, free, atoi, qsort
       - #include <string.h>     // for strlen, strcmp, strcpy, memset
       - #include <stdbool.h>    // for bool, true, false
       - #include <math.h>       // for sqrt, pow, floor, ceil
       - #include <limits.h>     // for INT_MAX, INT_MIN
       - #include <ctype.h>      // for isdigit, isalpha, tolower
       - #include <stdint.h>     // for uint64_t, int64_t

       Third-party headers:
       - #include <gmp.h>        // for arbitrary precision arithmetic (mpz_t)
       - #include "uthash.h"     // for hash tables (see usage below)

    2. DATA STRUCTURES:

       Hash Table (using uthash):
```c
       #include "uthash.h"
       
       // Define your hash entry structure
       struct hash_entry {
           int key;              // key
           int value;            // value
           UT_hash_handle hh;    // makes this structure hashable
       };
       
       struct hash_entry *hash = NULL;  // initialize to NULL
       
       // Add entry
       struct hash_entry *entry = malloc(sizeof(struct hash_entry));
       entry->key = some_key;
       entry->value = some_value;
       HASH_ADD_INT(hash, key, entry);
       
       // Find entry
       struct hash_entry *found;
       HASH_FIND_INT(hash, &key, found);
       if (found) {
           int val = found->value;
       }
       
       // Delete entry
       HASH_DEL(hash, entry);
       free(entry);
       
       // Delete all entries
       struct hash_entry *current, *tmp;
       HASH_ITER(hh, hash, current, tmp) {
           HASH_DEL(hash, current);
           free(current);
       }
```

       Dynamic Array (manual implementation):
```c
       int *arr = malloc(capacity * sizeof(int));
       // To resize:
       capacity *= 2;
       arr = realloc(arr, capacity * sizeof(int));
       // Remember to free:
       free(arr);
```

       Queue/Stack (manual implementation using arrays or linked lists)

       Big Integers (using GMP):
```c
       #include <gmp.h>
       
       mpz_t a, b, result;
       mpz_init(a);
       mpz_init(b);
       mpz_init(result);
       
       mpz_set_ui(a, 12345);           // set from unsigned int
       mpz_set_str(a, "123456789", 10); // set from string (base 10)
       
       mpz_add(result, a, b);           // result = a + b
       mpz_mul(result, a, b);           // result = a * b
       mpz_mod(result, a, b);           // result = a % b
       
       gmp_printf("%Zd\n", result);     // print
       
       mpz_clear(a);
       mpz_clear(b);
       mpz_clear(result);
```

    3. INPUT/OUTPUT FORMAT:
       - Input comes from STDIN using scanf()
       - Output goes to STDOUT using printf()
       - Read integers: scanf("%d", &n);
       - Read long: scanf("%lld", &n);
       - Read strings: char str[1000]; scanf("%s", str);
       - Read line: fgets(str, sizeof(str), stdin);
       - Print integer: printf("%d\n", result);
       - Print long: printf("%lld\n", result);
       - Print string: printf("%s\n", str);
       - Always add newline at the end of output
       - Match output format EXACTLY as specified in the problem

    4. MEMORY MANAGEMENT:
       - Always free dynamically allocated memory
       - uthash entries must be freed individually (see hash table section)
       - GMP variables must be cleared with mpz_clear()
       - Regular malloc() requires free()
       - Avoid memory leaks

    5. CODE STRUCTURE:
       - Write a complete, runnable C program
       - Always include a main() function that returns int
       - Return 0 from main() on success
       - Handle edge cases (empty input, boundary values, etc.)
       - Initialize all variables before use

    6. COMMON PATTERNS:

       Reading multiple integers:
```c
       int n;
       scanf("%d", &n);
       int *arr = malloc(n * sizeof(int));
       for (int i = 0; i < n; i++) {
           scanf("%d", &arr[i]);
       }
       free(arr);
```

       String processing:
```c
       char str[1000];
       scanf("%s", str);
       int len = strlen(str);
```

       Using hash table for counting:
```c
       struct hash_entry *hash = NULL;
       
       // Increment count for key
       struct hash_entry *found;
       HASH_FIND_INT(hash, &key, found);
       if (found) {
           found->value++;
       } else {
           struct hash_entry *new_entry = malloc(sizeof(struct hash_entry));
           new_entry->key = key;
           new_entry->value = 1;
           HASH_ADD_INT(hash, key, new_entry);
       }
```

       Sorting:
```c
       int compare(const void *a, const void *b) {
           return (*(int*)a - *(int*)b);
       }
       qsort(arr, n, sizeof(int), compare);
```

    7. ALGORITHM TYPES YOU MAY ENCOUNTER:
       - Array manipulation (search, sort, reverse, rotate)
       - String processing (parsing, pattern matching, transformations)
       - Hash tables (frequency counting, two-sum, anagrams)
       - Dynamic programming (memoization using hash tables)
       - Graph algorithms (BFS/DFS using manual queues/stacks)
       - Tree traversal (using recursion or manual queues)
       - Sliding window problems
       - Two pointers technique
       - Greedy algorithms
       - Mathematical computations (use GMP for big integers)
       - Number theory (modular arithmetic, primes, GCD/LCM)

    8. COMPILATION AND EXECUTION:
       - Your program will be compiled with: gcc -std=c11 -O2 -o program code.c -lm -lgmp
       - uthash.h is header-only, no linking needed
       - It will be run with test inputs via STDIN
       - Output will be compared character-by-character with expected output
       - Ensure output format matches exactly (spaces, newlines, etc.)

    9. EXAMPLE STRUCTURE:
```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <stdbool.h>
    #include <math.h>
    #include "uthash.h"

    struct hash_entry {
        int key;
        int value;
        UT_hash_handle hh;
    };

    int main() {
        // Read input
        int n;
        scanf("%d", &n);

        // Process using appropriate data structure
        struct hash_entry *map = NULL;

        // Your algorithm here
        for (int i = 0; i < n; i++) {
            // process
        }

        // Output result
        printf("%d\n", result);

        // Clean up
        struct hash_entry *current, *tmp;
        HASH_ITER(hh, map, current, tmp) {
            HASH_DEL(map, current);
            free(current);
        }

        return 0;
    }
```

    Remember: Write clean, efficient, and correct C code that solves the problem completely.
    Available libraries: standard C library, math.h (-lm), GMP (-lgmp), uthash.h (header-only)
"""
