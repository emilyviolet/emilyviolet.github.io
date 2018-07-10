# OpenMP internals, part 1: outlining
## What is OpenMP and why should I care?

## A high level overview
- Shared-memory multithreading library aimed at scientific computing (e.g. computational physics,
  weather simulation):
    * Relatively homogeneous thread execution. Contrast to e.g. web-server with one thread waiting on
      disk I/O, another listening to a socket, etc
    * Designed for high throughput/FLOPS in scientific applications
    * Quote three types of threading from: [](https://software.intel.com/en-us/articles/performance-obstacles-for-threading-how-do-they-affect-openmp-code)
    * We're aiming at doing the same work faster (generally don't get to choose which atoms to work on,
      so throughput is less of a concern) (I think)
    * Generally take some large workload (e.g. linear algebra) and split it relatively evenly between
      threads - one for each physical core (maybe see future post on HPC vs distributed systems?)
- All compiler directives:
    * Pragmas
    * Environment variables
    * Parallel regions and work-sharing constructs
    * Based on underlying OS thread primitives (e.g. POSIX pthreads) but adds high-level interfaces to
      make them easier to use
- Fork-join model:
    * Execution is serial until encountering a parallel region, then forks into threads which execute
      concurrently, before re-joining and continuing serially
    * Master thread (rank 0) and worker threads (rank 1 to Nthreads)
    * Thread pooling so less overhead from startup and tear-down
    * Thread teams - IDK very much about this
    * Definitely not going to go into too much detail or tutorial stuff about OpenMP, post some links
      instead

## Tools used in this series
- GDB doesn't seem to be much help, since it doesn't step into the #pragmas
    * Stack traces seem to work fine though.
    * Need to do
` (gdb) dir /usr/src/debug/gcc-7.3.1-2.fc27.x86_64/libgomp` to get it to find the source files.
- There has to be a way to get the intermediate code after the pragmas get processed though?
    * Nope, this all happens at the intermediate representation stage (sort of documented in the 
      [GCC internals guide](https://gcc.gnu.org/onlinedocs/gccint/OpenMP.html), so we can't just get the
      preprocessor output.
## Basic hello world:
```C   
    #include<stdio.h>                                                              
    #include<omp.h>                                                                
                                                                                   
    int main(){                                                                    
                                                                                   
        #pragma omp parallel                                                       
        {                                                                          
            int thread_num = omp_get_thread_num();                                 
            int num_threads = omp_get_num_threads();                               
            printf("Hello from thread %d of %d!\n", thread_num, num_threads);      
        }                                                                          
        return(0);                                                                 
    }
```
- Put a breakpoint at the first parallel instruction (line 8) and start execution
- Getting a stack-trace works okay:
```    Thread 1 "hello" hit Breakpoint 1, main._omp_fn.0 () at hello.c:8
8	        int thread_num = omp_get_thread_num();
(gdb) backtrace
#0  main._omp_fn.0 () at hello.c:8
#1  0x00007ffff7bb6cdf in GOMP_parallel (fn=0x40065b <main._omp_fn.0>, data=0x0, num_threads=4, flags=0)
    at ../../../libgomp/parallel.c:168
#2  0x0000000000400654 in main () at hello.c:4
```
- Our highest frame says we're at line 4 of hello.c: the start of the parallel region.
- We then move into an external function call `GOMP_parallel` in libgomp/parallel.c:
    * This introduces one of the fundamental abstractions that OpenMP provides - *outlining*
    * Whenever we encounter a parallel region, the compiler (GCC) transforms that region of code into a
      function with a corresponding data section (I'll get to that later when we talk about data sharing
      and scope), which then gets executed by each thread.

    * This high-level overview ignores a lot of the details. Specifically, how do we split the work-load
      between threads? How do we distribute data between threads? How do we schedule threads so any that
      finish early know to wait for all the other threads to finish?
- Finally, we execute our functions to print the current thread's ID and the total number of threads,
  which are stored internally in structs representing a thread and a *thread team*, which actually does
  the work (see below)


## Digging deeper into the source code
### GOMP_PARALLEL
- From the source code, we can see that `GOMP_parallel` takes a pointer to a function which takes a
  void pointer and returns void (i.e. it has no return value and takes a generic pointer to its
  arguments). It also takes a void pointer to some data (which is NULL in this case), the number of
  threads the region will be executed by and some optional flags (which aren't used in our *Hello World*
  example).
- Starts by determining the number of threads to use via `gomp_resolve_num_threads()`
    * The bare `#pragma omp parallel` in hello.c just uses the default/value of `OMP_NUM_THREADS`. This
      is stored in the task-specific *internal control variable* struct `gomp_task_icv->nthreads_var`,
      which also stores things like which scheduling scheme to use and whether to enable nested
      parallelism (documented in section 2.3.1 in 
      [the *OpenMP 4.0* specification](https://gcc.gnu.org/onlinedocs/gccint/OpenMP.html)).
    * Parallel regions can request a different number of threads, in which case
      `gomp_resolve_num_threads` has to do a bit more work to make sure we don't run with more than the
      maximum number of threads. Can also use a conditional clause like `#pragma omp parallel for
      if(cond)`, in which case the number of threads is forced to 1 if `cond` is false.
- Next, we start a *thread team*, then execute the body of the parallel section (via the `fn` function
  pointer):
    * Thread teams are teams of threads (imagine that!) which all execute some outlined function with a
      particular data environment. A thread team can have fewer than the maximum number of threads and
      there can be more than one thread team at a time (e.g. nested parallelism), provided the total
      number of threads in each team is less than the maximum number of threads (via `OMP_NUM_THREADS').
- Finally, we call `GOMP_parallel_end()` which cleans up the execution environment and releases the
  threads back to the pool, where they sit idle until needed.

## Thread Pooling
- Finally, let's look at one of the most useful features of OpenMP: thread pooling.
- Keeps idle threads in a logical *pool*: threads get created, do work, then return to the pool once
  they're done. Threads only need to be created once per program, then stay "alive" until the program is
  finished.

"
Finally, OpenMP has one more feature that we need to know about: thread pooling. As I said above,
creating and destroying threads can be quite expensive operations (at least on most CPUs), so we want to
do as little of these as possible. OpenMP handles this by keeping threads around in a *pool*, even when
they're not actively doing work. Threads get created upon entering the first parallel region of the
program, do some work, then get released back into the pool to sit idle until there's more work to do.
The upshot of this is that threads only get created and destroyed once - the overhead is spread out
across the programs whole runtime, at the cost of extra thread-management complexity. 
"

- Thread creation and destruction has some non-zero overhead (which needs to be done by the OS), so
  re-using threads between parallel sections gives significant performance gains.
- We can see this in action with GDB. Add an extra parallel region to hello.c:
``` C
int main(){
    
    printf("This is a serial region of code.\n");
    #pragma omp parallel
    {
        printf("Entering the first parallel region...\n");
        int thread_num = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        printf("Hello from thread %d of %d!\n", thread_num, num_threads);
    }
    
    #pragma omp parallel
    {
        printf("Entering the second parallel region...\n");
        int thread_num = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        printf("Hello from thread %d of %d!\n", thread_num, num_threads);
    }
    return(0);
}
```
- Running this with GDB shows that the threads only get created once, upon entering the first parallel
  region. The threads then get destroyed at the end of the program (which is **after** the last parallel
  region):
```
Starting program: /home/emily/Dropbox/Noodling/OpenMP/hello 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib64/libthread_db.so.1".
This is a serial region of code.
[New Thread 0x7ffff73d0700 (LWP 11156)]
[New Thread 0x7ffff6bcf700 (LWP 11157)]
[New Thread 0x7ffff63ce700 (LWP 11158)]
Entering the first parallel region...
Entering the first parallel region...
Entering the first parallel region...
Hello from thread 0 of 4!
Hello from thread 3 of 4!
Entering the first parallel region...
Hello from thread 2 of 4!
Hello from thread 1 of 4!
Entering the second parallel region...
Hello from thread 3 of 4!
Entering the second parallel region...
Hello from thread 2 of 4!
Entering the second parallel region...
Hello from thread 1 of 4!
Entering the second parallel region...
Hello from thread 0 of 4!
Finished all parallel regions.
[Thread 0x7ffff63ce700 (LWP 11158) exited]
[Thread 0x7ffff73d0700 (LWP 11156) exited]
[Thread 0x7ffff7fd1c00 (LWP 11152) exited]
[Inferior 1 (process 11152) exited normally]
```
- We can also see that OpenMP has an implicit barrier to synchronise threads at the end of a parallel
  regions. None of the threads enter the second parallel section until **all** of them have finished the
  first one.
