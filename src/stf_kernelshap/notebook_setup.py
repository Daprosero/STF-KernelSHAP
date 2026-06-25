"""Notebook environment setup for local, Colab, and Kaggle runs."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


DATASET_SLUG = "daprosero/mi-tdah-dataset"
DATASET_DIR_NAME = "MI_TDAH_Dataset"


@dataclass(frozen=True)
class NotebookPaths:
    repo_root: Path
    dataset_root: Path
    data_dir: Path
    models_dir: Path
    repo_models_dir: Path
    repo_results_dir: Path
    repo_figures_dir: Path
    output_models_dir: Path
    results_dir: Path
    figures_dir: Path
    temp_dir: Path
    debug: bool


def find_repo_root(start=None):
    """Find the repository root from a notebook or script working directory."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        has_notebooks_dir = (candidate / "Notebooks").exists() or (candidate / "notebooks").exists()
        if (candidate / "src").exists() and has_notebooks_dir:
            return candidate
    return current


def ensure_src_on_path(repo_root):
    src_path = str(Path(repo_root) / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _candidate_dataset_roots(repo_root, dataset_dir_name):
    repo_root = Path(repo_root)
    return [
        repo_root / dataset_dir_name,
        repo_root / "Data",
        repo_root,
    ]


def _has_dataset_layout(path):
    path = Path(path)
    return (
        (path / "Data").exists()
        or (path / "TDAH").exists()
        or (path / "MI").exists()
    )


def _normalize_dataset_root(path):
    path = Path(path).resolve()
    nested = path / DATASET_DIR_NAME
    if _has_dataset_layout(nested):
        return nested
    return path


def ensure_dataset(
    repo_root,
    dataset_slug=DATASET_SLUG,
    dataset_dir_name=DATASET_DIR_NAME,
    download_if_missing=True,
):
    """
    Resolve the MI/TDAH dataset for local, Colab, or Kaggle notebooks.

    Local preferred layout:
        <repo>/MI_TDAH_Dataset/Data/...

    Backward-compatible local layout:
        <repo>/Data/...

    If neither exists and kagglehub is available, the Kaggle dataset is
    downloaded and mirrored under <repo>/MI_TDAH_Dataset.
    """
    repo_root = Path(repo_root).resolve()

    for candidate in _candidate_dataset_roots(repo_root, dataset_dir_name):
        if _has_dataset_layout(candidate):
            return _normalize_dataset_root(candidate)

    if not download_if_missing:
        raise FileNotFoundError(
            f"No dataset folder found. Expected {repo_root / dataset_dir_name} "
            f"or {repo_root / 'Data'}."
        )

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "Dataset not found locally and kagglehub is not installed. "
            "Install kagglehub or place the dataset under "
            f"{repo_root / dataset_dir_name}."
        ) from exc

    downloaded_path = Path(kagglehub.dataset_download(dataset_slug)).resolve()
    target_path = repo_root / dataset_dir_name

    if not target_path.exists():
        try:
            os.symlink(downloaded_path, target_path, target_is_directory=True)
        except OSError:
            shutil.copytree(downloaded_path, target_path, dirs_exist_ok=True)

    print("Path to dataset files:", downloaded_path)
    return _normalize_dataset_root(target_path)


def _resolve_data_dir(dataset_root):
    dataset_root = Path(dataset_root)
    if (dataset_root / "Data").exists():
        return dataset_root / "Data"
    return dataset_root


def _resolve_models_dir(repo_root, dataset_root):
    repo_root = Path(repo_root)
    dataset_root = Path(dataset_root)
    if (dataset_root / "Models").exists():
        return dataset_root / "Models"
    return repo_root / "Models"


def setup_notebook_environment(debug=False, download_dataset=True):
    """
    Configure paths for notebooks.

    debug=False:
        Outputs go to Results/, Figures/, Models/ in the cloned repo.

    debug=True:
        Outputs go to Temp/Results, Temp/Figures, Temp/Models in the repo.
    """
    repo_root = find_repo_root()
    ensure_src_on_path(repo_root)
    dataset_root = ensure_dataset(repo_root, download_if_missing=download_dataset)

    temp_dir = repo_root / "Temp"
    output_root = temp_dir if debug else repo_root

    paths = NotebookPaths(
        repo_root=repo_root,
        dataset_root=dataset_root,
        data_dir=_resolve_data_dir(dataset_root),
        models_dir=_resolve_models_dir(repo_root, dataset_root),
        repo_models_dir=repo_root / "Models",
        repo_results_dir=repo_root / "Results",
        repo_figures_dir=repo_root / "Figures",
        output_models_dir=output_root / "Models",
        results_dir=output_root / "Results",
        figures_dir=output_root / "Figures",
        temp_dir=temp_dir,
        debug=debug,
    )

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.output_models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository root: {paths.repo_root}")
    print(f"Dataset root: {paths.dataset_root}")
    print(f"Debug mode: {paths.debug}")
    print(f"Results dir: {paths.results_dir}")
    print(f"Figures dir: {paths.figures_dir}")
    print(f"Output models dir: {paths.output_models_dir}")

    return paths
