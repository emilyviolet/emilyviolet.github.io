---
title: "OpenMP internals part 1: Outlining and parallel execution"
author: "Emily Kahl"
---

## Please rise for a message from the author
Wow that was a long hiatus - my last blog post was like, two years ago. I've 
been really lax in making content due to a combination of Real Life Events and
the fact that PhDs are *really* time consuming (who'd have thought?). But I've
finally got some space to focus on fun stuff, so let's dive back into my 
favourite topic: esoteric computing stuff!

## Introduction and tools

[Last time](https://emilyviolet.github.io/2018/07/10/ompenmp-introduction.html), we had a brief look at
through what OpenMP is and roughly what it does, as well as a few gripes I have about things it doesn't
do well. I strongly recommend reading that post if you haven't already (and haven't used OpenMP before),
as this one won't make a lot of sense without that background knowledge. 

In this post, we're going to dip our toes into the internals of OpenMP and take a look what how a
parallel program actually executes. All of this post will be at a relatively high-level compared with
what's to come, but we *will* be using gdb ([the GNU debugger](https://www.gnu.org/software/gdb/), which
I'll refer to in lower-case throughout the rest of this post) to
track what's going on inside a C program as it executes. I'll try to explain things as we go, but at
least some familiarity with C and gdb would be handy as you follow along. 
[This tutorial](http://www.cs.toronto.edu/~krueger/csc209h/tut/gdb_tutorial.html) is a good place to
start if you've never used gdb or need a refresher.

Just like in [part 0](https://emilyviolet.github.io/2018/07/10/ompenmp-introduction.html), I'll
be using gcc's OpenMP implementation, called `libgomp`. I had considered using LLVM, since its internals
are generally much better documented, but I find the `libgomp` source code to much more readable so we're
sticking with gcc for now.

Also heads up, this post is going to be very UNIX-specific (technically Linux-specific, for the
nitpickers). *Most* of the concepts in this post should map reasonably well to Windows and other fruity 
operating systems, but I know UNIX so that's what I'm writing about.

## Tracing with gdb
Let's start with the *Hello World* program from last time:

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

As a reminder, if you compile this program with `gcc -fopenmp -o hello hello.c`, you should get output
similar to:

```
Hello from thread 0 of 4!
Hello from thread 2 of 4!
Hello from thread 1 of 4!
Hello from thread 3 of 4!
```

Where the number of "Hello"s will depend on the number of cores your CPU has.

So far, so good. Now, let's compile the program with debugging information via the `-g` compiler flag
and run it through gdb. What we're going to do is put a *break-point* at the call to
`omp_get_thread_num()` (on line 8 of the source code) so that gdb will stop the program's execution
just inside the parallel region. Then we should be able to use gdb's tools to inspect the program's
state and get a bit of an idea of what OpenMP is actually doing.

The commands used to execute the above process should look something like (`$` prompts indicate shell
commands, `(gdb)` prompts indicate commands to run in gdb):

```
$ gdb ./hello
(gdb) break 10
(gdb) run
```

(I've elided a bunch of output from gdb here for brevity's sake).

Once we've hit the break-point, we can get a stack-trace to see how the program got to that point:

```    
Thread 1 "hello" hit Breakpoint 1, main._omp_fn.0 () at hello.c:10
10	        printf("Hello from thread %d of %d!\n", thread_num, num_threads);
(gdb) backtrace
#0  main._omp_fn.0 () at hello.c:10
#1  0x00007ffff7bb6cdf in GOMP_parallel (fn=0x40065b <main._omp_fn.0>, data=0x0, num_threads=4, flags=0)
    at ../../../libgomp/parallel.c:168
#2  0x0000000000400654 in main () at hello.c:4
```

(Your output might look slightly different, as the breakpoints and stack frames will have different
memory addresses, but the important parts of the output should be identical across Unix systems.)

Let's go through the output piece-by-piece:

- Our highest frame at #2 says we're at line 4 of hello.c: the start of `main()`.
- We then move (frame #1) into an external function call `GOMP_parallel` in libgomp/parallel.c. This stack 
frame introduces one of the fundamental abstractions that OpenMP provides - *outlining*. Whenever we 
encounter a parallel region, the compiler (gcc) transforms, or outlines, that region of code into a 
function with a corresponding data section (I'll get to that later when we talk about data sharing and 
scope), which then gets executed by the program's threads.

    * This high-level overview ignores a lot of the details. Specifically, it says nothing about how we 
      split the work-load between threads, as well as data sharing and scheduling. But we'll get to that
      soon.
      
- Finally at frame #0, we execute our functions to print the current thread's ID and the total number of
threads, which are stored internally in structs representing a thread and a *thread team*, which 
actually does the work (more on that below). The line of code we actual wrote in `main()` is actually
wrapped in a function `main._omp_fn.0`, which is the actual code executed by each thread in the parallel
region.

Even though we have essentially no insight into the internal workings of the steps I've outlined above,
gdb still gives us a good high-level overview of how OpenMP achieves its parallelism. Indeed, the basic
steps of outlining our section of code, spinning up a parallel environment (including threads, data
structures and scheduling policies), and then passing the outlined functions to each thread applies to  
almost every parallel application works with GNU OpenMP. As we'll see, this 
neatly encapsulates the way that we can translate high-level `#pragmas` into the low-level
primitives that an operating system actually understands.

## The OpenMP source code
gdb is an extremely powerful tool for understanding what a particular program is *actually* doing when
it executes (in fact, I'd argue that it's one of the most useful things that any programmer can learn),
and has already shown us more about the internals of OpenMP than I've learned from any textbook. But we
need to go deeper if we want to truly grok what's going on, not just in terms of the nuts-and-bolts of
threading, but also more heady topics like "why is it designed this way?". gdb can only take us so
far by itself (but we'll still make extensive use of it throughout this series), so let's crack 
open the source code!

### GOMP_parallel
The function we're interested in here is called `GOMP_parallel()`, which is defined in the source file
`parallel.c`. If you'd like to follow along or poke around yourself, you can download the gcc source code 
from the [gcc mirror on GitHub](https://github.com/gcc-mirror/gcc) (just don't try to use that repo to 
submit changes, the Gnu project is a bit behind the times). 

Now, let's go line-by-line through the actual body of the function, which I have copy-pasted below:

```C
void                                                                           
GOMP_parallel (void (*fn) (void *), void *data, unsigned num_threads, unsigned int flags)                                                 
{                                                                              
  num_threads = gomp_resolve_num_threads (num_threads, 0);                     
  gomp_team_start (fn, data, num_threads, flags, gomp_new_team (num_threads), NULL);                                                              
  fn (data);                                                                   
  ialias_call (GOMP_parallel_end) ();                                          
}  
```

Starting with the function signature, we can see that `GOMP_parallel` takes a pointer to a function 
which in turn takes a void pointer as its argument, as well as a pointer to some data, which is
NULL in our hello world example since we're not actually using any variables in the parallel section.
These first two arguments define our parallel workload: `*fn` is the actual work that we want our
threads to execute, which is derived from outlining the `omp parallel` section. The `*data` pointer is a
little bit more tricky, since it needs to handle all the variables which are transferred between the
serial and parallel regions, which it stores in a custom struct data type.

Data movement requires a fair bit of work on the part of the compiler in order to work transparently, 
as it needs
to distribute variables between threads and (in the case of the special *lastprivate* storage class) 
ensure that values are appropriately synchronised and transferred out of the parallel region once it has
finished. This is a somewhat convoluted way of what boils down to invoking a function, 
but it maps fairly closely to how threading works in both the Linux kernel (via `pthreads`) and
Windows. Basically, the OS kernel doesn't understand fancy high-level 
scheduling and memory models; the kernel understands pointers and functions 
(which are passed around as pointers), so OpenMP needs to translate our
directives into something the OS can work with.

Lastly, `GOMP_parallel` also takes in 
the number of threads to use when executing the region and some optional flags 
(which aren't used in our *Hello World* example). 

Now, let's look at the actual body of the function:
- We start by determining the number of threads to use via `gomp_resolve_num_threads()`
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
  pointer, derived from the outlined section of our *Hello World* example):
    * Thread teams are teams of threads (imagine that!) which all execute some outlined function with a
      particular data environment. A thread team can have fewer than the maximum number of threads and
      there can be more than one thread team at a time (e.g. nested parallelism), provided the total
      number of threads in each team is less than the maximum number of threads (via `OMP_NUM_THREADS').
- Finally, we call `GOMP_parallel_end()` which cleans up the execution environment and releases the
  threads back to the pool, where they sit idle until needed.

Pretty straightforward on the surface, but there's two details I want to drill
down on. The first is the question of just how OpenMP generates the outlined 
function, which will really need its own separate blog post to dig through all
the details (next week, fingers crossed). The second is thread creation and 
pooling; it's important, but we can get a rough idea by poking around with gcc.
Let's do that now.

## Thread Pooling

Creating and destroying threads is handled by the operating system kernel and 
has a nonzero cost in CPU time (especially for the 
relatively heavyweight OpenMP threads), so we want to
do as little of this as possible. OpenMP handles this by keeping threads 
around in a *pool*, so they're kept "alive" even
not actively doing work. Threads get created upon entering the first parallel region of the
program, do some work, then get released back into the pool to sit idle until there's more work to do.
The upshot of this is that threads only get created and destroyed once - the overhead is spread out
across the program's whole runtime, at the cost of extra thread-management 
complexity when entering or leaving a parallel region. Most of the time, this
tradeoff is worthwhile, since a lot of the thread management can be handled at
compile time. But as we'll see later on, certain dynamic workload distribution
patterns can cause the thread management overhead to really tank the
performance.

We can see a simple example of thread-pooling in action by modifying our 
little hello world program and inspecting its execution with gdb. Let's add an
extra parallel region to hello.c:
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
If you run this under gdb, you should get an output that looks something like
the following:

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

We can see a couple of neat things just from this toy example. First, the 
threads only get created once, upon entering the first parallel region. The 
threads then persist through the second region and are destroyed at the end of
the program (which, again, is **after** the last parallel region).

We can also infer that OpenMP has an implicit barrier to synchronise threads 
at the end of a parallel regions, since none of the threads enter the second 
parallel section until **all** of them have finished the first one. This
inference is born out by checking the source code: in the listing of 
`GOMP_parallel`, we can see that `GOMP_parallel` finishes with a call to 
`GOMP_parallel_end`, which is also defined in `libgomp/parallel.c`. The actual
source definition is a bit fiddly so I won't include it here (although I
encourage you to look it up if you're interested), but it does indeed check to
make sure there are no more busy threads and waits before leaving the parallel
region.

The thread synchronisation is
really important for code correctness; you don't want the main body of the
code to try to access some data the threads haven't finished generating yet.
It's also important to be aware of the performance implications - if one 
thread takes longer to finish its work then all the remaining threads will be 
stuck twiddling their metaphorical thumbs at the end of the parallel region 
until the slowpoke is finished. It's very convoluted to try to implement manual
barriers to let the program can carry on with any future instructions that
don't yet depend on the outcome of the parallel region, so it's a bit trickier
to do effective balancing for complicated workloads. 

This is not just a hypothetical performance problem either - a lot of the 
challenges in optimising (AMBiT)[https://github.com/drjuls/AMBiT] came from the
somewhat limited options OpenMP provides for handling inherently unbalanced 
workloads. I'll probably touch on this in some later post, but for now just
know that thread synchronisation and balancing are Hard Problems that crop up
on basically any non-trivial workload.


## Wrapping up
Let's summarise what we've learned so far:
1. OpenMP implements its parallelism by *outlining* - transforming the contents
   of a parallel region into its own separate function, which it then passes
   to low-level threading primitives.
2. Variables are passed in and out of parallel regions by sticking them all
   in a struct, and passing a pointer to the thread primitive. We still haven't
   seen what this looks like in a real workload, but can see the broad outlines
   just by analysing the source code and tracing our program with gdb.
3. OpenMP creates threads at the beginning of the first parallel section in a
   program and then keeps them around in a *pool* until the program finishes
   its execution. Creating and destroying threads is computationally expensive,
   so OpenMP tries to do it as infrequently as possible.

This is of course a wild oversimplification, but it's good enough for a rough 
introduction to the basic concepts. There's a few things I've glossed over,
which I want to dig into in the next few posts:
1. How does the compiler actually go about outlining a parallel region?
2. What does an outlined function look like?
3. How is work assigned to threads? How does OpenMP manage to keep idle threads
   ``alive'', and still manage to pass new workloads to them?

In order to answer the first two points, we're going to need to dig into the 
innards of gcc and learn a little bit about the nitty-gritty of compilation. 
I'm pretty dang excited about this - I've always been interested in the theory
and design of compilers, as well as practical implementation concerns, but 
never had a good opportunity to really dig in and learn how they work. The last
point will come a little bit later and it'll require a *lot* more digging
through the `libgomp` source code. I hope you'll join me; it's going to be fun!
