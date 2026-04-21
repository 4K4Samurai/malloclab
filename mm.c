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
    char padding[4]; /* offset 4, padding to align to pointers */
    struct free_block *next; /* offset 8 */
    struct free_block *prev; /* offset 16 */
};

/* Basic constants and macros */
#define WSIZE       sizeof(struct boundary_tag)  /* Word and header/footer size (bytes) */
#define MIN_BLOCK_SIZE_WORDS 8  /* Minimum block size in words */
#define CHUNKSIZE  (1<<10)  /* Extend heap by this amount (words) */

static inline size_t max(size_t x, size_t y) {
    return x > y ? x : y;
}

static size_t align(size_t size) {
  return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

static bool is_aligned(size_t size) __attribute__((__unused__));
static bool is_aligned(size_t size) {
  return size % ALIGNMENT == 0;
}

static size_t round_up_to_alignment_words(size_t words)
{
    return (words + 7) & ~((size_t)7);
}

/* Global variables */
static struct block *heap_listp = 0;  /* Pointer to first block */  

static struct free_block free_list_head; /* this should be sentinel, so it is never allocated */

/* Function prototypes for internal helper routines */
static struct block *extend_heap(size_t words);
static void place(struct block *bp, size_t asize);
static struct block *find_fit(size_t asize);
static struct block *coalesce(struct block *bp);

/* Given a block, obtain previous's block footer.
   Works for left-most block also. */
static struct boundary_tag * prev_blk_footer(struct block *blk) {
    return &blk->header - 1;
}

/* Return if block is free */
static bool blk_free(struct block *blk) { 
    return !blk->header.inuse; 
}

/* Return size of block is free */
static size_t blk_size(struct block *blk) { 
    return blk->header.size; 
}

/* Given a block, obtain pointer to previous block.
   Not meaningful for left-most block. */
static struct block *prev_blk(struct block *blk) {
    struct boundary_tag *prevfooter = prev_blk_footer(blk);
    assert(prevfooter->size != 0);
    return (struct block *)((void *)blk - WSIZE * prevfooter->size);
}

/* Given a block, obtain pointer to next block.
   Not meaningful for right-most block. */
static struct block *next_blk(struct block *blk) {
    assert(blk_size(blk) != 0);
    return (struct block *)((void *)blk + WSIZE * blk->header.size);
}

/* Given a block, obtain its footer boundary tag */
static struct boundary_tag * get_footer(struct block *blk) {
    return ((void *)blk + WSIZE * blk->header.size)
                   - sizeof(struct boundary_tag);
}

/* Set a block's size and inuse bit in header and footer */
static void set_header_and_footer(struct block *blk, int size, int inuse) {
    blk->header.inuse = inuse;
    blk->header.size = size;
    * get_footer(blk) = blk->header;    /* Copy header to footer */
}

/* Mark a block as used and set its size. */
static void mark_block_used(struct block *blk, int size) {
    set_header_and_footer(blk, size, 1);
}

/* Mark a block as free and set its size. */
static void mark_block_free(struct block *blk, int size) {
    set_header_and_footer(blk, size, 0);
}

/* 
 * mm_init - Initialize the memory manager 
 */
int mm_init(void) 
{
    free_list_head.next = &free_list_head;
    free_list_head.prev = &free_list_head;

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
    initial[2] = FENCE;                     /* Prologue footer */
    heap_listp = (struct block *)&initial[3];
    initial[3] = FENCE;                     /* Epilogue header */

    /* Extend the empty heap with a free block of CHUNKSIZE bytes */
    if (extend_heap(CHUNKSIZE) == NULL) 
        return -1;
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
 * This will insert a block at the front of the explicit free list.
 * Then casts the block to a free_block to access the next/prev pointers.
 */
static void fl_insert(struct block *blk)
{
    struct free_block *fb = (struct free_block *)blk;
    fb->next = free_list_head.next; /* New block points forward to old first */
    fb->prev = &free_list_head; /* New block points back to the sentinel */
    free_list_head.next->prev = fb; /* Old first points back to the new block */
    free_list_head.next = fb; /* Sentinel points forward to the new block */ 
}

/*
 * Ideas and code were derived from Claude.
 * This will remove a block from the explicit free list.
 * Then casts the block to a free_block to access next/prev pointers.
 * Then splices the block out by linking it to a predecessor directly to the
 * successor. 
 */
static void fl_remove(struct block *blk)
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
    bool prev_alloc = prev_blk_footer(bp)->inuse;   /* is previous block allocated? */
    bool next_alloc = ! blk_free(next_blk(bp));     /* is next block allocated? */
    size_t size = blk_size(bp);

    if (!next_alloc)
    {
        fl_remove(next_blk(bp));
        size = size + blk_size(next_blk(bp));
    }

    if (!prev_alloc)
    {
        size_t prev_size = blk_size(prev_blk(bp));
        fl_remove(prev_blk(bp));
        bp = prev_blk(bp);
        size = size + prev_size;
    }

    mark_block_free(bp, size);
    fl_insert(bp);

    return bp;
}

/*
 * mm_realloc - Naive implementation of realloc
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

    void *newptr = mm_malloc(size);

    /* If realloc() fails the original block is left untouched  */
    if (!newptr) {
        return 0;
    }

    /* Copy the old data. */
    struct block *oldblock = ptr - offsetof(struct block, payload);
    size_t oldpayloadsize = blk_size(oldblock) * WSIZE - 2 * sizeof(struct boundary_tag);
    if (size < oldpayloadsize) oldpayloadsize = size;
    memcpy(newptr, ptr, oldpayloadsize);

    /* Free the old block. */
    mm_free(ptr);

    return newptr;
}

/* 
 * checkheap - We don't check anything right now. 
 */
void mm_checkheap(int verbose)
{ 
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
 * find_fit - Find a fit for a block with asize words 
 * Ideas and code were derived from Claude.
 */
static struct block *find_fit(size_t asize)
{
    /* First fit search */
    for (struct free_block *fp = free_list_head.next;
        fp != &free_list_head; fp = fp->next)
        {
            if (asize <= blk_size((struct block *)fp))
            {
                return (struct block *)fp;
            }
        }
    return NULL; /* No fit */
}

team_t team = {
    /* Team name */
    "Sample allocator using implicit lists",
    /* First member's full name */
    "Keiji Cabanacan",
    "kgcabanacan@vt.edu",
    /* Second member's full name (leave as empty strings if none) */
    "Dinmukhamed Shabarov",
    "dinmukhamed@vt.edu",
};