# NoIPCA Documentation Index

This directory contains all documentation for the NoIPCA project.

## Essential Documentation

### Getting Started

📘 **[README.md](README.md)** - Complete project documentation
- Installation and setup
- Usage examples and API reference
- Architecture overview

📋 **[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)** - Step-by-step workflow guide
- How to generate panels
- How to compute factors
- How to analyze results

📁 **[DIRECTORY_STRUCTURE.txt](DIRECTORY_STRUCTURE.txt)** - Complete directory structure
- File organization
- Module descriptions
- Import structure

### Changes from Original

📝 **[CHANGES.md](CHANGES.md)** - Complete summary of all changes from original codebase
- Directory structure changes
- Code organization improvements
- Functional changes and bug fixes
- Performance optimizations
- Breaking changes and migration guide

### Performance & Optimization

🚀 **[ACCELERATION_SUMMARY.md](ACCELERATION_SUMMARY.md)** - Performance improvements overview
- Speedup metrics and benchmarks
- Numba acceleration (3-5x speedups)
- Randomized SVD ridge regression (20-100x for large D)
- Implementation details

🚀 **[ACCELERATION_QUICKSTART.md](ACCELERATION_QUICKSTART.md)** - Quick guide to optimization features
- How to enable Numba acceleration
- When randomized SVD kicks in
- Performance configuration

🚀 **[ACCELERATION_D10000.md](ACCELERATION_D10000.md)** - High-dimensional optimization (D=10,000)
- Randomized SVD deep dive
- Production scale performance
- Configuration for large-scale problems

---

## Quick Navigation

### By Task

**Want to run the code?**
→ Start with [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)

**Want to understand the structure?**
→ See [DIRECTORY_STRUCTURE.txt](DIRECTORY_STRUCTURE.txt)

**Want to know what changed from original?**
→ Read [CHANGES.md](CHANGES.md)

**Want to optimize performance?**
→ Check [ACCELERATION_SUMMARY.md](ACCELERATION_SUMMARY.md)

### By Role

**End Users:**
1. [README.md](README.md) - Overview and installation
2. [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) - How to use
3. [ACCELERATION_QUICKSTART.md](ACCELERATION_QUICKSTART.md) - Performance tips

**Developers:**
1. [DIRECTORY_STRUCTURE.txt](DIRECTORY_STRUCTURE.txt) - Code organization
2. [CHANGES.md](CHANGES.md) - What's different from original
3. [ACCELERATION_SUMMARY.md](ACCELERATION_SUMMARY.md) - Performance details
4. `../utils/README.md` - General utilities documentation
5. `../utils_factors/README.md` - Factor utilities documentation

**Researchers:**
1. [CHANGES.md](CHANGES.md) - Methodological differences
2. [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) - Reproducibility guide
3. [ACCELERATION_D10000.md](ACCELERATION_D10000.md) - Large-scale implementation

---

## File Organization

```
docs/
├── README.md                    # Main documentation
├── INDEX.md                     # This file
├── WORKFLOW_GUIDE.md            # Usage guide
├── DIRECTORY_STRUCTURE.txt      # Structure reference
├── CHANGES.md                   # Changes from original (NEW - consolidated)
├── ACCELERATION_SUMMARY.md      # Performance overview
├── ACCELERATION_QUICKSTART.md   # Quick performance guide
└── ACCELERATION_D10000.md       # Large-scale optimization
```

All intermediate change documents have been consolidated into [CHANGES.md](CHANGES.md).

---

## Additional Resources

### Within Package

- **`../utils/README.md`** - General utilities (ridge regression, standardization, etc.)
- **`../utils_factors/README.md`** - Factor computation utilities (Fama, DKKM, portfolio stats)
- **`../utils_bgn/README.md`** - BGN model documentation
- **`../utils_kp14/README.md`** - KP14 model documentation
- **`../utils_gs21/README.md`** - GS21 model documentation
- **`../tests/README.md`** - Test suite documentation

### External

- Original paper: [Reference to paper if available]
- GitHub repository: [If applicable]

---

## Documentation Updates

**Last consolidated:** December 2024

All intermediate change documents (reorganization summaries, bug fix summaries, optimization tracking) have been consolidated into a single comprehensive [CHANGES.md](CHANGES.md) document that explains the current state versus the original codebase.
