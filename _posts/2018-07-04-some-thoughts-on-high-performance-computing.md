# Some thoughts on high-performance computing
My research group recently open-sourced our kickass atomic structure code 
[AMBiT](https://github.com/drjuls/AMBiT). AMBiT is my baby - I spent the last year and a bit getting it
fit for production, which is a bit less than half of my PhD. Along with the usual bug hunting and
documentation, I spent the bulk of that time overhauling AMBiT's parallelism and tracking down
performance pathologies to better utilise modern high-performance computing (HPC) architectures.

Our paper is still working it's way through peer-review (although a preprint is up on the 
[arXiv](https://arxiv.org/abs/1805.11265)), but in the mean-time I thought I'd
use this blog to talk about a few interesting things I've learned throughout the process. Most of this
is going to focus on performance issues and strange quirks of parallel programming, since that's what I
spent the bulk of my time on.

I'm sort of hoping this post can provide a decent overview of what I've been working on and why it's
interesting, as well as clear up some terminology I'll be using later on. Specifically, there seems to
be a lot of misunderstanding around scientific and high-performance computing, to the point that people
often mean vastly different things when they talk about HPC. While I can't promise to solve these
disputes forever, I do want to make sure that we're all on the same page before I start going into more
depth.

## What even is high-performance computing?
High-performance computing is one of those nebulous terms that means different things to different
fields of computer science. I sometimes hear it applied to things like large-scale distributed systems
and data centres, which makes a certain degree of sense - cloud infrastructure does need to be
highly performant, after all. But, data centres and the like tend to be more loosely-coupled running a
heterogeneous set of I/O-heavy jobs (serving queries, running distributed database instances and the
like), which is pretty strongly not what I do. I tend to deal with [tightly-coupled, homogeneous
clusters](https://insidehpc.com/hpc101/hpc-architecture-for-beginners/)
(supercomputers, if I want to sound all fancy and high-tech) running relatively few huge numerical jobs
in parallel. My kind of problems also tend to be strongly CPU- and memory-bound, which is qualitatively
different to the kind of work they do at Amazon or Google (this is a broad generalisation, obviously).

My kind of work also frequently gets lumped in with "scientific computing", another similarly nebulous
term. And while my work is technically scientific computing, the broader *category* of scientific
computing also includes things like lightweight statistical modelling or tiny python implementations of
some toy physical model, which aren't really high-performance. Again, the term is close, but doesn't
quite reflect the particular challenges of what I work on.

When I talk about HPC, I pretty much always mean the "high-performance" subset of scientific computing.
It's the huge numerical modelling problems that come out of atomic physics, computational fluid
dynamics or weather forecasting. The kind of numerical calculations that need weeks or months of solid
number-crunching spread across an enormous number of cores on massively parallel supercomputers to even
have a chance at solving. This combination of huge problem size, CPU- and memory-bound workloads and
large-scale parallelism pretty much defines HPC (at least in my books). While I'm not knocking other 
fields of computer science, HPC does present a set of extremely interesting and difficult problems to
overcome. The biggest one is performance, so let's talk about my performance constraints.

## Putting the P in HPC
One common piece of wisdom in software engineering circles is that developer time is far more valuable
than compute time. Most of the time this is very sound advice; if you're building a small web app then
it's definitely not worth spending heaps of developer time trying to shave off a couple of percent
overhead. HPC is different though. We use AMBiT every day and calculations can take anywhere from hours
to days (or even weeks in one case), so even a 5% performance improvement can add up to huge gains in
productivity. 

And this isn't even that huge by computational physics standards! It's not uncommon for huge projects
like molecular dynamics simulations or lattice QCD (simulating quarks and gluons, the fundamental
particles which make up the familiar protons and neutrons) to take **months** of compute time on massive
supercomputers. Seen in this light, it often *does* make sense to spend several months tracking down
and fixing performance pathologies. And it *can* take months to track down some of these problems,
especially since HPC requires reasoning about massive-scale parallelism, which is notoriously difficult 
to get right.

As I see it, HPC software development has two chief values: performance and correctness. That's it.
Sure, HPC people care about things like security and usability, but we're much more willing to
make trade-offs and sacrifice them for performance and correctness. To put it bluntly, performance is
the reason Fortran is still incredibly common in my field, despite it's name being [more or less
synonymous with unreadable spaghetti code](https://queue.acm.org/detail.cfm?id=1039535).

I suspect the narrow set of values, as well as the incredible strictness of the performance constraints,
is largely responsible for the impedance mismatch whenever I talk to programmers from other fields about
my work. For example, I've had multiple well-meaning programmers ask "why don't you just use a nice,
high-level language like Python or Haskell?" It's not an unreasonable question, since they're coming
from a domain with different values (usually things like maintainability, security or 
rapid product turnaround time). As I said before, it would be amazingly wasteful for your average web
developer to muck around with low-level things like [CPU cache 
friendliness](https://stackoverflow.com/questions/16699247/what-is-a-cache-friendly-code); there are far
more pressing things to worry about. (Note: I don't mean to pick on web developers here. Like any field,
there are good and bad webdevs, but the domain is just about as far away from scientific computing as
you can get).

For HPC though, Python and Haskell are total non-starters. We actually *are* sensitive to tiny
effects like cache misses (which is only around 60ns on my university's cluster), operating system
context switches and overhead for multithreading via [OpenMP](https://en.wikipedia.org/wiki/OpenMP)
(which I'll be spending a lot of time on in future posts). *This* is why C and Fortran dominate
scientific computing: we simply can't afford even moderate run-time overheads, let alone something like
the CPython interpreter. Reasoning about performance in this context is extremely difficult, so it
should come as no surprise that I've learned a **lot** about performance and HPC architecture over the
course of my PhD.

## Making AMBiT snappy
At 15 years old, AMBiT is fairly young by atomic physics standards - the last project I worked on was at
least 20 years old, but there were sections of the codebase dating back to the 80s, at least. That being
said, my supervisor is a very strong programmer and had been more or less consistently developing AMBiT
for that whole 15 years, so it was already fairly well optimised by the time I got my hands on it.
This made performance engineering somewhat tricky, since basically all of the low-hanging fruit (stuff
like loop ordering, stray quadratic algorithms and so on) had already been taken care of. There was, one
reasonably large target that I could focus on, though: parallelism. 

AMBiT originally only made use of
[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface), a library which provides a set of
machine-independent interfaces for passing *messages* (conceptually similar to packets) between
tightly-coupled processes on HPC architecture. The basic idea is that MPI allows you to spin up multiple
processes running the same executable on a cluster, then abstracts away the details of the communication
and synchronisation between them. The processes can reside across multiple *nodes* (basically,
independent computers connected together via low-latency, high-bandwidth ethernet connections), with
possibly multiple processes per node (ideally with one process per physical core). The same set of MPI
functions handle communication *within* nodes (via shared memory) and *between* nodes (via the network
interface) depending on where the processes get spun up. It's all very cool engineering and is the
de facto standard for HPC applications.

The problem is, our algorithm imposed a per-process memory overhead which grew with the problem size.
The net result being that AMBiT required more memory as you added more MPI processes, and we'd rapidly
exhaust the available memory if we used more than about 8 processes per node. This was really bad, since
the remaining cores on each node we used were just left sitting idle; we were effectively leaving half
of our hardware sitting around twiddling it's thumbs. 

There were two possible solutions at the time:

* Try and overhaul the algorithm to reduce the per-process memory overhead: this would be a huge job
  with no guarantee of success. I could very well have spent the rest of my PhD fiddling around with
  numerical methods and have nothing to show for it come thesis time.
* Switch to a hybrid model of parallelism with OpenMP: OpenMP is a shared memory multithreading
  interface specifically tuned for HPC applications,
  so we'd only need to spin up one MPI process per node, then use multiple threads to keep all of the
  node's cores busy. This way, each node gets a smaller, fixed overhead which is totally independent of
  the number of cores we use.

We went with the second option, and even though it took way longer than expected, I think the hybrid
MPI+OpenMP approach was ultimately the better option in the long run. AMBiT is now able to scale up to
arbitrarily large calculations to a degree of accuracy that I didn't even think was *possible* when I
started my PhD. Seriously, our demo calculation in [the paper](https://arxiv.org/abs/1805.11265) was of
the spectrum of singly-ionised chromium: an extremely complicated ion with five-valence electrons
(atomic structure calculations generally grow exponentially with the number of valence-electrons and
even five is a lot more than most codes can handle). It is, to my knowledge, the most accurate
calculation of Cr+ yet published, and we did it as a *side-effect* of publishing our code. So yeah, AMBiT
kicks a lot of ass.

## Lessons learned
Porting a mature codebase to a totally different parallel programming paradigm is a maelstrom of
suffering and madness, with OpenMP responsible for the majority of the madness. On top of the usual
difficulties with multithreading and concurrency (race-conditions and lock-contention spring immediately
to mind), the internal workings of OpenMP are pretty badly documented, so there's lots of fun and
surprising ways to tank your program's performance if you're not careful. And boy did I hit a lot of
them.

Despite all of this, I think OpenMP is a really valuable tool for HPC. It has a relatively friendly
interface based around compiler directives, which abstracts away a lot of the work of dealing with
threads and (in theory) allows the programmer to focus on their *algorithm* rather than fussing around
with raw `pthreads` or whatever. 

In addition to sharing what I've learned about OpenMP this last year, I'd like to try and fill in some
of the gaps in the literature around its inner workings. There's a lot of high-level tutorials on how to
*use* OpenMP, but even the best, most detailed ones don't have anything to say about how it actually
works. Like, what does the compiler *actually do* when I tell it to parallelise some section of code? I
haven't been able to find a comprehensive, but accessible guide to this, so I figure I might as well
write one myself. 

To that end (and also because it's super interesting!), I've started teaching myself the inner workings
of OpenMP. I'm still technically a physicist (even though, deep down, my heart belongs to computer
science) so I probably can't put any of this in my thesis. But! There's no reason I can't write this
stuff down here, so I'm making a series of posts detailing the inner workings of OpenMP, specifically,
GCC's `libgomp' implementation of it. I've found the stuff I've learned so far to be super interesting,
hopefully you will too!
