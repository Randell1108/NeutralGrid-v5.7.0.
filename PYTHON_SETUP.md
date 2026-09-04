# Python Setup Guide

## Python Version

This project uses **Python 3.12.8** (installed at: `C:\Users\cris_\AppData\Local\Programs\Python\Python312`)

Python 3.14 has been removed from the system.

## Running the Application

To run the application, simply use:

```bash
python main.py
```

Or to run with uvicorn:

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You can also use the `py` launcher:

```bash
py main.py
```

## Installing Additional Packages

To install additional packages:

```bash
python -m pip install <package-name>
```

Or with the py launcher:

```bash
py -m pip install <package-name>
```

## Installed Packages

All packages from `requirements.txt` have been successfully installed:

- ✓ fastapi 0.128.0
- ✓ uvicorn 0.40.0
- ✓ pydantic 2.12.5
- ✓ python-binance 1.0.34
- ✓ numpy 2.4.1
- ✓ pandas 2.3.3
- ✓ scipy 1.17.0
- ✓ scikit-learn 1.8.0
- ✓ **hmmlearn 0.3.3** (previously failed on Python 3.14)
- ✓ openpyxl 3.1.5
- ✓ aiosqlite 0.22.1
- ✓ python-dotenv 1.2.1
- ✓ aiohttp 3.13.3

## Why Python 3.12?

Python 3.12 was chosen over 3.14 because:
1. Better package ecosystem support
2. Pre-built wheels available for all required packages (including hmmlearn)
3. No need for Microsoft Visual C++ Build Tools
4. Stable and widely supported by scientific computing libraries

Python 3.14 was removed from the system to avoid conflicts and ensure consistency.
