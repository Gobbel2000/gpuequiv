# gpuequiv

[Documentation](https://docs.rs/gpuequiv) |
[Crates.io](https://crates.io/crates/gpuequiv)

A GPU-accelerated implementation of an algorithm to find all equivalences of
processes in the Linear-time Branching-time Spectrum.

This project is part of my bachelor's thesis at Technische Universität Berlin.
The accompanying thesis text is hosted at
[Gobbel2000/thesis-gpuequiv](https://github.com/Gobbel2000/thesis-gpuequiv).

This is a Rust crate implementing the
[Spectroscopy algorithm](https://arxiv.org/abs/2303.08904) by B. Bisping
with a focus on performance and scalability.
To this end, the most critical parts of the algorithm are accelerated by GPU
compute shaders.
The [wgpu](https://github.com/gfx-rs/wgpu) crate is used
for interfacing with the system's GPU API.
Shaders are written in the WebGPU Shading Language
([WGSL](https://gpuweb.github.io/gpuweb/wgsl/)).
These technologies are based on the up-and-coming WebGPU API,
aiming to enable advanced access to GPUs in web browsers.
Therefore, this crate can be used in WebAssembly,
although it requires the browser to support the WebGPU API.

Requires **Rust version** >= 1.73.

### Equivalences

For an input Labeled Transition System, the algorithm will decide for any
process pair which of the behavioral equivalences in the following spectrum
hold and which don't:

* Enabledness
* Trace Equivalence
* Failures
* Failure Traces
* Readiness
* Readiness Traces
* Revivals
* Impossible Futures
* Possible Futures
* Simulation
* Ready Simulation
* Nested 2-Simulation
* Bisimulation

## Examples

The `examples` directory contains a few files showcasing some of the ways this
crate can be used. This command runs the examples:

```sh
cargo run --example NAME
```

where `NAME` is the name of one of the examples, for example `compare_all`.
Additional logging can be enabled using the `RUST_LOG` environment variable.
For example:

```sh
RUST_LOG=gpuequiv=debug,info cargo run --example full
```

### WebAssembly

WASM is explicitly supported.
This requires a browser that supports the WebGPU API, see the current
[implementation status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status).

The example `compare_all` is configured to be tested on WASM. Simply run:

```sh
cargo run-wasm
```

and then open [localhost:8000](http://localhost:8000). The output should appear
in the browser console.


## Tests

Unit and integration tests can be run with:

```sh
cargo test
```


### Benchmarks

Execute benchmarks with:

```sh
cargo bench

```

This runs the algorithm on part of the VLTS (Very Large Transition System)
benchmark suite. Benchmarks are controlled by
[Criterion-rs](https://github.com/bheisler/criterion.rs).
