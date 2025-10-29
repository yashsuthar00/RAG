# RAG
Retrieval-Augmented Generation (RAG) demo — simple pipeline to ingest a PDF, create embeddings
and store them in MongoDB Atlas, and run similarity queries against the vector store.

## Requirements

- Recommended Python: 3.10 or 3.11 (some ML libraries may not yet provide wheels for very new
	CPython releases). If you run into build/wheel errors, create a venv with Python 3.11.
- A MongoDB Atlas cluster (or a locally running MongoDB) and a connection string with write access.
- A `cv/cv.pdf` file in the repository (the ingestion script reads `./cv/cv.pdf`).
- The project includes `requirements.txt` with the expected dependencies. Use it to install exact
	versions.

## Quick setup (Windows PowerShell)
Open VS Code terminal (PowerShell) in the project root `C:\Users\yashs\Desktop\yash\RAG` and run the following steps.

1. Create and activate a virtual environment

```powershell
python -m venv .venv
# If PowerShell prevents script execution, enable per-user RemoteSigned once (see troubleshooting)
.\.venv\Scripts\Activate.ps1
```

If activation fails due to execution policy, run this once then re-open the terminal:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
.\.venv\Scripts\Activate.ps1
```

Alternative (without activating): call the venv Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe ingestion.py
```

2. Upgrade pip and install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you prefer a single install command that fetches commonly-needed packages (in case `requirements.txt`
is missing or you want the latest):

```powershell
python -m pip install python-dotenv langchain langchain-community langchain-mongodb pymongo sentence-transformers transformers huggingface-hub pypdf
```

### Install from requirements.txt (explicit)

If you already created and activated the virtual environment, install all pinned dependencies from the
provided `requirements.txt` like this:

```powershell
# (venv activated)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you don't want to (or can't) activate the venv, call the venv Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If a package fails to install due to binary wheel issues (common for `torch` or other heavy packages),
consider using Python 3.11 or following the package-specific installation instructions (for example,
installing `torch` from the official PyTorch site which provides compatible wheels for Windows).

3. Create the `.env` file with your MongoDB connection string

Create a file named `.env` in the project root (or set the environment variable directly). At minimum add:

```text
CONNECTION_STRING="your-mongodb-connection-string"
```

Notes about `.env` and code paths:
- The scripts call `load_dotenv('/app/.env')` which expects `/app/.env` (a container path). On Windows you can either:
	- create a file at that path and update the script, or
	- edit the scripts (`ingestion.py`, `demo.py`, `retriver.py`) to use `load_dotenv()` or `load_dotenv('.env')`, or
	- set the `CONNECTION_STRING` environment variable in your shell instead of using a `.env` file.

Example: keep `ingestion.py` unchanged and set the environment variable in PowerShell for the current session:

```powershell
$env:CONNECTION_STRING = 'your-mongodb-connection-string'
python ingestion.py
```

4. Ensure `cv/cv.pdf` exists

Place the PDF you want to index at `cv/cv.pdf`. The script filters out very short pages.

5. Run the ingestion script

```powershell
# (venv activated)
python ingestion.py
```

This will:
- read and split the `cv.pdf` into chunks,
- call the metadata tagger and a Hugging Face LLM (the repo id in the code is `HuggingFaceH4/zephyr-7b-beta`),
- compute embeddings and store vector documents into the MongoDB collection configured by `CONNECTION_STRING`.

6. Run example query code

```powershell
python demo.py
# or
python retriver.py
```

## Troubleshooting

- Execution policy errors when activating the venv: run the `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force` command once and restart the terminal.
- If pip fails to find a package (e.g. `langchain-mongodb`) check the exact PyPI package name or use `pip search` / the package docs.
- If you get binary wheel / build errors (commonly for `torch`, `sentence-transformers`):
	- Prefer Python 3.11 or 3.10, and install `torch` using the official instructions from PyTorch (they provide prebuilt wheels for Windows). Example for CUDA/CPU options available at https://pytorch.org/get-started/locally/.
- If the scripts attempt to use a large LLM from Hugging Face you may need more RAM or to change the model to a small CPU-friendly option.

## Save the environment

Once everything is working, freeze your environment for reproducibility:

```powershell
python -m pip freeze > requirements.txt
```

## Security and credentials

- Do not commit `.env` or any credentials to source control. `.gitignore` already excludes `.env` and `.venv`.
- Use least-privilege MongoDB credentials for the connection string used by these scripts.

## Final notes

If you paste any error output you encounter after following these steps (ModuleNotFoundError, connection errors, or runtime exceptions), I can help you fix the next issue.

Happy experimenting!

