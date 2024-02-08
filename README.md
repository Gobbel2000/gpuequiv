# gpuequiv

A GPU-accelerated implementation of an algorithm to find all equivalences of
processes in the Linear-time Branching-time Spectrum.

This project is part of my bachelor's thesis at Technische Universität Berlin.
The accompanying thesis text is hosted at
[Gobbel2000/thesis-gpuequiv](https://github.com/Gobbel2000/thesis-gpuequiv).

This is a Rust library implementing the
[Spectroscopy algorithm](https://arxiv.org/abs/2303.08904) by B. Bisping
with a focus on performance and scalability.
To this end, the most critical parts of the algorithm are accelerated by GPU
compute shaders.
The [wgpu](https://github.com/gfx-rs/wgpu) library is used
for interfacing with the system's GPU API.
Shaders are written in the WebGPU Shading Language
([WGSL](https://gpuweb.github.io/gpuweb/wgsl/)).
These technologies are based on the up-and-coming WebGPU API,
aiming to enable advanced access to GPUs in web browsers.
Therefore, this library can be used in WebAssembly,
although it requires the browser to support the WebGPU API.

Requires **rust version** >= 1.73.

## Examples

The `examples` directory contains a few files showcasing some of the ways this
library can be used. This command runs the examples:

```sh
cargo run --example NAME
```

where `NAME` is the name of one of the examples, for example `compare_all`.
Additional logging can be enabled using the `RUST_LOG` environment variable.
For example:

```sh
RUST_LOG=gpuequiv=debug,info cargo run --example full
```

The example `compare_all` can be run on WASM. Simply run:

```sh
cargo run-wasm
```

and then open [localhost:8000](http://localhost:8000). The output should appear
in the browser console. This requires a browser that supports the WebGPU API,
see the current
[implementation status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status).


## Tests

Unit and integration tests can be run with:

```sh
cargo test
```
