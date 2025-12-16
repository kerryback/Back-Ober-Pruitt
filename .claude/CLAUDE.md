# Claude Code Guidelines for This Project

## Platform: Windows

This project runs on **Windows**, which has specific requirements and limitations that must be considered when writing code.

## Critical Windows Issues

### 1. Console Encoding (cp1252)

**NEVER use Unicode box-drawing or special characters in print statements.**

❌ **DO NOT USE:**
```python
print(f"  {'─'*50}")  # U+2500 box drawing character
print(f"  {'━'*50}")  # U+2501 heavy box drawing
print(f"  {'•'*50}")  # U+2022 bullet
print(f"  {'…'}")      # U+2026 horizontal ellipsis
```

✅ **ALWAYS USE:**
```python
print(f"  {'-'*50}")  # ASCII hyphen-minus
print(f"  {'='*50}")  # ASCII equals
print(f"  {'*'*50}")  # ASCII asterisk
print(f"  {'...'}")   # ASCII periods
```

**Reason:** Windows console uses cp1252 encoding by default, which cannot encode Unicode characters beyond the basic ASCII + Latin-1 range. This causes `UnicodeEncodeError` exceptions.

### 2. File Paths

- Always use `os.path.join()` or `pathlib.Path` for cross-platform compatibility
- Windows uses backslashes `\` but Python accepts forward slashes `/`
- Use raw strings `r"path\to\file"` or escape backslashes `"path\\to\\file"`

### 3. Line Endings

- Windows uses CRLF (`\r\n`) line endings
- Git may show warnings about LF being replaced with CRLF - this is normal
- Use `.gitattributes` to control line ending behavior if needed

### 4. Shell Commands

- When using `subprocess` or `Bash` tool:
  - Quote paths with spaces: `"C:\Program Files\..."`
  - Use `&&` to chain commands, not `;` or newlines
  - Be aware that some Unix commands don't exist on Windows

## Project-Specific Notes

### Parallel Processing

- This project uses `joblib.Parallel` with `n_jobs=10` for parallel processing
- The SDF compute modules (BGN, KP14, GS21) do NOT use nested parallelization
- `calculate_moments.py` processes in chunks to manage memory

### File Naming Convention

- Arrays: `{model}_{index}_arrays.pkl`
- Moments: `{model}_{index}_moments.pkl`
- Factors: `{model}_{index}_fama.pkl`, `{model}_{index}_dkkm_{nfeatures}.pkl`

### Performance Characteristics

- Later chunks in `calculate_moments.py` take longer due to increasing historical data
- This is expected behavior - computation at month `t` requires all data from `0` to `t`

## Common Errors to Avoid

1. **UnicodeEncodeError** - Use only ASCII characters in print statements
2. **Path errors** - Always use `os.path.join()` or forward slashes
3. **Encoding errors in file I/O** - Specify `encoding='utf-8'` when writing text files
4. **Line ending warnings** - Normal on Windows, can be ignored

## Testing on Windows

When making changes:
- Test all print statements work in Windows console
- Verify file paths work with Windows backslashes
- Check that subprocess commands work in Windows environment
