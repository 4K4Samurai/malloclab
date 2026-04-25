/* 
 * Simple, 64-bit allocator based on implicit free lists, 
 * first fit placement, and boundary tag coalescing, as described
 * in the CS:APP2e text. Blocks must be aligned to 16 byte
 * boundaries. Minimum block size is 16 bytes. 
 *
 * This version is loosely based on 
 * http://csapp.cs.cmu.edu/3e/ics3/code/vm/malloc/mm.c
 * but unlike the book's version, it does not use C preprocessor 
 * macros or explicit bit operations.
 *
 * It follows the book in counting in units of 4-byte words,
 * but note that this is a choice (my actual solution chooses
 * to count everything in bytes instead.)
 *
 * You should use this code as a starting point for your 
 * implementation.
 *
 * First adapted for CS3214 Summer 2020 by gback
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

#include "mm.h"
#include "memlib.h"
#include "config.h"

struct boundary_tag {
    int inuse:1;        // inuse bit
    int size:31;        // size of block, in words
                        // block size
};

/* FENCE is used for heap prologue/epilogue. */
const struct boundary_tag FENCE = {
    .inuse = -1,
    .size = 0
};

/* A C struct describing the beginning of each block. 
 * For implicit lists, used and free blocks have the same 
 * structure, so one struct will suffice for this example.
 *
 * If each block is aligned at 12 mod 16, each payload will
 * be aligned at 0 mod 16.
 */
struct block {
    struct boundary_tag header; /* offset 0, at address 12 mod 16 */
    char payload[0]; /* offset 4, at address 0 mod 16 */
};

/* Ideas were derived from claude */
struct free_block {
    struct boundary_tag header; /* offset at 0, at address 12 mod 16 */
    char padding[4]; /* offset 4: header is 4 bytes, pointers are 8-byte aligned on x86-64 */
    struct free_block *next; /* offset 8 */
    struct free_block *prev; /* offset 16 */
};

/* Basic constants and macros */
#define WSIZE       sizeof(struct boundary_tag)  /* Word and header/footer size (bytes) */
#define MIN_BLOCK_SIZE_WORDS 8  /* header(1) + padding(1) + next(2) + prev(2) + footer(1) = 7 words, rounded to 8 */
#define CHUNKSIZE  (1<<10)  /* Extend heap by this amount (words) */

static inline size_t max(size_t x, size_t y) {
    return x > y ? x : y;
}

/* Round size up to the next power of 2 if it falls in the top 1/8 of that
 * class (e.g. 448→512). Freed blocks from one allocation cycle can then be
 * reused by a slightly-larger same-class request, cutting external fragmentation
 * on traces with near-power-of-2 size patterns. */
static inline size_t size_class_round(size_t size) {
    if (size > 512) return size;
    size_t power = 8;
    while (power < size) power <<= 1;
    return (size >= power - power / 8) ? power : size;
}

static inline size_t align(size_t size) {
  return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

static bool is_aligned(size_t size) __attribute__((__unused__));
static bool is_aligned(size_t size) {
  return size % ALIGNMENT == 0;
}

/* Round up to a multiple of 4 words (16 bytes).
 * Blocks start at 12 mod 16, so payload lands at 0 mod 16 for any 4-word-multiple block size. */
static inline size_t round_up_to_alignment_words(size_t words)
{
    return (words + 3) & ~((size_t)3);
}

/* Segregated free list: 12 size classes (boundaries in words):
 *   0:≤8  1:≤16  2:≤32  3:≤64  4:≤128  5:≤256  6:≤512
 *   7:≤1024  8:≤2048  9:≤4096  10:≤8192  11:>8192
 * Finer large-block granularity gives near-best-fit for random-size traces
 * without paying O(n) best-fit cost. */
#define NUM_CLASSES 12
static struct free_block free_lists[NUM_CLASSES];

static inline int size_class(size_t words) {
    if (words <=    8) return 0;
    if (words <=   16) return 1;
    if (words <=   32) return 2;
    if (words <=   64) return 3;
    if (words <=  128) return 4;
    if (words <=  256) return 5;
    if (words <=  512) return 6;
    if (words <= 1024) return 7;
    if (words <= 2048) return 8;
    if (words <= 4096) return 9;
    if (words <= 8192) return 10;
    return 11;
}

/* Global variables */
static struct block *heap_listp = 0;  /* Pointer to first block */

/* Function prototypes for internal helper routines */
static struct block *extend_heap(size_t words);
static void place(struct block *bp, size_t asize);
static struct block *find_fit(size_t asize);
static struct block *coalesce(struct block *bp);

/* Given a block, obtain previous's block footer.
   Works for left-most block also. */
static inline struct boundary_tag * prev_blk_footer(struct block *blk) {
    return &blk->header - 1;
}

/* Return if block is free */
static inline bool blk_free(struct block *blk) {
    return !blk->header.inuse;
}

/* Return size of block in words */
static inline size_t blk_size(struct block *blk) {
    return blk->header.size;
}

/* Given a block, obtain pointer to previous block.
   Not meaningful for left-most block. */
static inline struct block *prev_blk(struct block *blk) {
    struct boundary_tag *prevfooter = prev_blk_footer(blk);
    assert(prevfooter->size != 0);
    return (struct block *)((void *)blk - WSIZE * prevfooter->size);
}

/* Given a block, obtain pointer to next block.
   Not meaningful for right-most block. */
static inline struct block *next_blk(struct block *blk) {
    assert(blk_size(blk) != 0);
    return (struct block *)((void *)blk + WSIZE * blk->header.size);
}

/* Given a block, obtain its footer boundary tag */
static inline struct boundary_tag * get_footer(struct block *blk) {
    return ((void *)blk + WSIZE * blk->header.size)
                   - sizeof(struct boundary_tag);
}

/* Set a block's size and inuse bit in header and footer */
static inline void set_header_and_footer(struct block *blk, int size, int inuse) {
    blk->header.inuse = inuse;
    blk->header.size = size;
    * get_footer(blk) = blk->header;    /* Copy header to footer */
}

/* Mark a block as used and set its size. */
static inline void mark_block_used(struct block *blk, int size) {
    set_header_and_footer(blk, size, 1);
}

/* Mark a block as free and set its size. */
static inline void mark_block_free(struct block *blk, int size) {
    set_header_and_footer(blk, size, 0);
}

/* 
 * mm_init - Initialize the memory manager 
 */
int mm_init(void) 
{
    for (int i = 0; i < NUM_CLASSES; i++) {
        free_lists[i].next = &free_lists[i];
        free_lists[i].prev = &free_lists[i];
    }

    assert (sizeof(struct boundary_tag) == 4);

    /* Create the initial empty heap */
    struct boundary_tag * initial = mem_sbrk(4 * sizeof(struct boundary_tag));
    if (initial == NULL)
        return -1;

    /* We use a slightly different strategy than suggested in the book.
     * Rather than placing a min-sized prologue block at the beginning
     * of the heap, we simply place two fences.
     * The consequence is that coalesce() must call prev_blk_footer()
     * and not prev_blk() because prev_blk() cannot be called on the
     * left-most block.
     */
    /* initial[0..1] are unused alignment padding before the prologue fence. */
    initial[2] = FENCE;                     /* Prologue footer */
    heap_listp = (struct block *)&initial[3];
    initial[3] = FENCE;                     /* Epilogue header */

    return 0;
}

/* 
 * mm_malloc - Allocate a block with at least size bytes of payload 
 */
void *mm_malloc(size_t size)
{
    struct block *bp;      

    /* Ignore spurious requests */
    if (size == 0)
        return NULL;

    size = size_class_round(size);

    /* Adjust block size to include overhead and alignment reqs. */
    size_t bsize = align(size + 2 * sizeof(struct boundary_tag));    /* account for tags */
    if (bsize < size)
        return NULL;    /* integer overflow */
    
    /* Adjusted block size in words */
    size_t awords = max(MIN_BLOCK_SIZE_WORDS, round_up_to_alignment_words(bsize/WSIZE)); /* respect minimum size */

    /* Search the free list for a fit */
    if ((bp = find_fit(awords)) != NULL) {
        place(bp, awords);
        return bp->payload;
    }

    /* No fit found. Get more memory and place the block */
    size_t extendwords = max(awords,CHUNKSIZE); /* Amount to extend heap if no fit */
    if ((bp = extend_heap(extendwords)) == NULL)  
        return NULL;

    place(bp, awords);
    return bp->payload;
} 

/* 
 * mm_free - Free a block 
 */
void mm_free(void *bp)
{
    assert (heap_listp != 0);       // assert that mm_init was called
    if (bp == 0) 
        return;

    /* Find block from user pointer */
    struct block *blk = bp - offsetof(struct block, payload);

    mark_block_free(blk, blk_size(blk));
    coalesce(blk);
}

/*
 * Ideas and code were derived from Claude.
 * This will insert a block at the front of the appropriate size-class free list.
 */
static inline void fl_insert(struct block *blk)
{
    struct free_block *fb = (struct free_block *)blk;
    struct free_block *head = &free_lists[size_class(blk_size(blk))];
    fb->next = head->next;
    fb->prev = head;
    head->next->prev = fb;
    head->next = fb;
}

/* Insert at tail of the largest class. Realloc remainders (end_free) must
 * land in class7 at TAIL regardless of their actual size so that small
 * mallocs (which search classes 0→6 first) never steal end_free before
 * they reach the stranded block, preserving in-place expansion adjacency. */
static inline void fl_insert_tail(struct block *blk)
{
    struct free_block *fb = (struct free_block *)blk;
    struct free_block *head = &free_lists[NUM_CLASSES - 1];
    fb->prev = head->prev;
    fb->next = head;
    head->prev->next = fb;
    head->prev = fb;
}

/*
 * Ideas and code were derived from Claude.
 * This will remove a block from the explicit free list.
 * Then casts the block to a free_block to access next/prev pointers.
 * Then splices the block out by linking it to a predecessor directly to the
 * successor. 
 */
static inline void fl_remove(struct block *blk)
{
    struct free_block *fb = (struct free_block *)blk;
    fb->prev->next = fb->next; /* Predecessor skips over the free block */
    fb->next->prev = fb->prev; /* Successor points back to the predecessor */
}

/*
 * coalesce - Boundary tag coalescing. Return ptr to coalesced block
 * Ideas and code were derived from Claude 
 */
static struct block *coalesce(struct block *bp)
{
    struct block *next = next_blk(bp);
    bool prev_alloc = prev_blk_footer(bp)->inuse;   /* is previous block allocated? */
    bool next_alloc = !blk_free(next);              /* is next block allocated? */
    size_t size = blk_size(bp);

    /* Merge next before prev: next_blk(bp) is only valid while bp hasn't moved yet. */
    if (!next_alloc)
    {
        fl_remove(next);
        size += blk_size(next);
    }

    if (!prev_alloc)
    {
        struct block *prev = prev_blk(bp);
        fl_remove(prev);
        size += blk_size(prev);
        bp = prev;
    }

    mark_block_free(bp, size);
    fl_insert(bp);

    return bp;
}

/*
 * Ideas and code were derived from Claude.
 * mm_realloc - Now resizes an allocated block to at least a number of size of bytes.
 * This will attempt to in-place expansion by absorbing a free next block if it is large enough.
 * Then will fall back to malloc/memcpy/free if the expansion is not possible.
 */
void *mm_realloc(void *ptr, size_t size)
{
    /* If size == 0 then this is just free, and we return NULL. */
    if (size == 0) {
        mm_free(ptr);
        return 0;
    }

    /* If oldptr is NULL, then this is just malloc. */
    if (ptr == NULL) {
        return mm_malloc(size);
    }

    struct block *blk = ptr - offsetof(struct block, payload);
    size_t oldsize = blk_size(blk);

    /* Need to compute the required block size in words, this includes the header and footer */
    size_t bsize = align(size + 2 * sizeof(struct boundary_tag));
    size_t awords = max(MIN_BLOCK_SIZE_WORDS, round_up_to_alignment_words(bsize / WSIZE));

    /* If the current block is already large enough, shrink and split if excess is usable */
    if (awords <= oldsize)
    {
        if ((oldsize - awords) >= MIN_BLOCK_SIZE_WORDS) {
            mark_block_used(blk, awords);
            struct block *remainder = next_blk(blk);
            mark_block_free(remainder, oldsize - awords);
            coalesce(remainder);  /* coalesce handles fl_insert */
        }
        return ptr;
    }

    struct block *next = next_blk(blk);

    /* If the next block is free and the combined size is sufficient, expand in place */
    if (blk_free(next) && (oldsize + blk_size(next)) >= awords)
    {
        size_t merged = oldsize + blk_size(next);
        fl_remove(next);
        if ((merged - awords) >= MIN_BLOCK_SIZE_WORDS) {
            /* Split: return excess to free list rather than holding it */
            mark_block_used(blk, awords);
            struct block *remainder = next_blk(blk);
            mark_block_free(remainder, merged - awords);
            fl_insert_tail(remainder);
        } else {
            mark_block_used(blk, merged);
        }
        return ptr;
    }

    /* If at the end of the heap, extend in place — avoids malloc+copy+free entirely */
    if (blk_size(next) == 0) {
        size_t need = awords - oldsize;
        if (mem_sbrk(need * WSIZE) == NULL) return NULL;
        mark_block_used(blk, awords);
        next_blk(blk)->header = FENCE;
        return ptr;
    }

    /* Fallback: find a free block (or extend heap), place at tail so the
     * remainder isn't stolen by the next small allocation, enabling the
     * following realloc to expand in-place instead of falling back again. */
    struct block *newblk = find_fit(awords);
    if (newblk == NULL) {
        newblk = extend_heap(max(awords, CHUNKSIZE));
        if (newblk == NULL) return NULL;
    }
    size_t newcsize = blk_size(newblk);
    fl_remove(newblk);
    if ((newcsize - awords) >= MIN_BLOCK_SIZE_WORDS) {
        mark_block_used(newblk, awords);
        struct block *remainder = next_blk(newblk);
        mark_block_free(remainder, newcsize - awords);
        fl_insert_tail(remainder);
    } else {
        mark_block_used(newblk, newcsize);
    }

    size_t copysize = oldsize * WSIZE - 2 * sizeof(struct boundary_tag);
    if (size < copysize) copysize = size;
    memcpy(newblk->payload, ptr, copysize);
    mm_free(ptr);
    return newblk->payload;
}

/*
 * checkheap - Verify heap invariants.
 */
void mm_checkheap(int verbose)
{
    /* Walk every block and check header/footer consistency and alignment */
    for (struct block *blk = heap_listp; blk_size(blk) > 0; blk = next_blk(blk)) {
        struct boundary_tag *footer = get_footer(blk);
        if (blk->header.inuse != footer->inuse || blk->header.size != footer->size) {
            fprintf(stderr, "checkheap: header/footer mismatch at %p\n", (void *)blk);
            abort();
        }
        if (!is_aligned((size_t)blk->payload)) {
            fprintf(stderr, "checkheap: payload misaligned at %p\n", (void *)blk);
            abort();
        }
        /* No two consecutive free blocks should exist (coalescing invariant) */
        if (blk_free(blk) && blk_size(next_blk(blk)) > 0 && blk_free(next_blk(blk))) {
            fprintf(stderr, "checkheap: consecutive free blocks at %p\n", (void *)blk);
            abort();
        }
    }

    /* Every block in each free list must be free and have consistent pointers.
     * Note: fl_insert_tail places realloc remainders in class7 regardless of
     * their actual size, so we don't check size_class(blk) == cls here. */
    for (int cls = 0; cls < NUM_CLASSES; cls++) {
        for (struct free_block *fp = free_lists[cls].next;
             fp != &free_lists[cls]; fp = fp->next) {
            if (!blk_free((struct block *)fp)) {
                fprintf(stderr, "checkheap: allocated block on free list[%d] at %p\n",
                        cls, (void *)fp);
                abort();
            }
            if (fp->next->prev != fp || fp->prev->next != fp) {
                fprintf(stderr, "checkheap: free list pointer inconsistency at %p\n", (void *)fp);
                abort();
            }
        }
    }

    if (verbose)
        fprintf(stderr, "checkheap: OK\n");
}

/* 
 * The remaining routines are internal helper routines 
 */

/* 
 * extend_heap - Extend heap with free block and return its block pointer
 */
static struct block *extend_heap(size_t words)
{
    void *bp = mem_sbrk(words * WSIZE);

    if (bp == NULL)
        return NULL;

    /* Initialize free block header/footer and the epilogue header.
     * Note that we overwrite the previous epilogue here. */
    struct block * blk = bp - sizeof(FENCE);
    mark_block_free(blk, words);
    next_blk(blk)->header = FENCE;

    /* Coalesce if the previous block was free */
    return coalesce(blk);
}

/* 
 * place - Place block of asize words at start of free block bp 
 *         and split if remainder would be at least minimum block size
 * Ideas and code were derived from Claude
 */
static void place(struct block *bp, size_t asize)
{
    size_t csize = blk_size(bp);
    fl_remove(bp);

    if ((csize - asize) >= MIN_BLOCK_SIZE_WORDS) {
        mark_block_used(bp, asize);
        struct block *remainder = next_blk(bp);
        mark_block_free(remainder, csize - asize);
        fl_insert(remainder);
    }
    else
    {
        mark_block_used(bp, csize);
    }
}

/*
 * find_fit - Find a fit for a block with asize words.
 * Searches size classes from the smallest that could hold asize upward,
 * giving near-best-fit behavior without scanning the whole free list.
 */
static struct block *find_fit(size_t asize)
{
    for (int cls = size_class(asize); cls < NUM_CLASSES; cls++) {
        for (struct free_block *fp = free_lists[cls].next;
             fp != &free_lists[cls]; fp = fp->next) {
            if (asize <= blk_size((struct block *)fp))
                return (struct block *)fp;
        }
    }
    return NULL;
}

team_t team = {
    /* Team name */
    "Keiji and Dimash",
    /* First member's full name */
    "Keiji Cabanacan",
    "kgcabanacan@vt.edu",
    /* Second member's full name (leave as empty strings if none) */
    "Dinmukhamed Shabarov",
    "dinmukhamed@vt.edu",
};