# Developer Workflow

This document details how to develop, run, and test code examples and book content in this repository.

## Prerequisites

- **Bazel**: The build and execution system. You only need Bazel installed on your system. All other tools (including `mdbook` and compiler toolchains) are managed hermetically by Bazel.

---

## 1. Running the Book

All `mdbook` commands are wrapped via Bazel targets.

### Start the Local Development Server
To serve the book with hot-reloading (default port `3000`):
```bash
bazel run //:serve
```

### Build the Static HTML
To generate the static HTML pages in the `book/` directory:
```bash
bazel run //:build
```

### Clean Build Artifacts
```bash
bazel run //:mdbook -- clean
```

---

## 2. Writing Chapters

1. **Add Markdown Pages**: Create your page (e.g., `src/gpu_optimization.md`).
2. **Register in Table of Contents**: Add your file to [src/SUMMARY.md](file:///home/yusuke/workspace/optimizing-machine-learning/src/SUMMARY.md):
   ```markdown
   - [GPU Optimization](gpu_optimization.md)
   ```

---

## 3. Adding and Including Code Examples

We keep code examples in the `examples/` directory.

### Step 1: Create the Source Code
Create your implementation and a corresponding test under `examples/<name>/`:
- `examples/quantization/quantize.cc`
- `examples/quantization/quantize_test.cc`

### Step 2: Define Bazel Targets
Create `examples/quantization/BUILD.bazel`:
```bazel
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "quantize",
    srcs = ["quantize.cc"],
)

cc_test(
    name = "quantize_test",
    srcs = ["quantize_test.cc"],
    deps = [":quantize"],
)
```

### Step 3: Embed Code inside Book Chapters
In your chapter markdown page (e.g., `src/gpu_optimization.md`), dynamically include the source files:
```markdown
Here is the implementation of our quantization helper:

\`\`\`cpp
{{#include ../examples/quantization/quantize.cc}}
\`\`\`
```
*Note: The relative path is always evaluated relative to the `src/` directory.*

---

## 4. Testing and Verification

To ensure all code examples are compilable, correctly functioning, and up-to-date:
```bash
# Run all unit tests in the workspace
bazel test //...

# Rebuild the mdBook
bazel run //:build
```
