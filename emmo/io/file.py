"""Module used to define functions for common I/O operations on files."""
import inspect
import json
import os
import pickle
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Callable
from typing import DefaultDict
from typing import TextIO
from typing import Union

import numpy as np
import pandas as pd
import yaml
from cloudpathlib import anypath
from cloudpathlib import CloudPath


COMPRESSION2EXTENSION = {"gzip": ".gz", "bz2": ".bz2", "zip": ".zip", "xz": ".xz", "zstd": ".zst"}

Openable = Union[str, Path, CloudPath]


def get_num_lines(file_path: Openable, remove_header: bool) -> int:
    """Get the number of lines in a file.

    Args:
        file_path: Local or remote path to the file.
        remove_header: Indicates if we should remove the header.
            This is useful in case of a CSV / TSV file.
    """
    num_lines = _load_file(file_path=file_path, load_fn=lambda f: sum(1 for _ in f), mode="r")

    return num_lines - remove_header


def load_txt(file_path: Openable) -> list[str]:
    """Load a local / remote .txt file.

    Args:
        file_path: Local or remote .txt file to load.

    Returns:
        List with each item corresponding to a line of the file.

    Raises:
        FileNotFoundError: The `file_path` provided doesn't exist.
    """
    return _load_file(
        file_path=file_path,
        load_fn=lambda f: [item.strip() for item in f],
        mode="r",
    )


def save_txt(
    data: list,
    file_path: Openable,
    force: bool = False,
    use_background_thread: bool = False,
) -> None:
    """Save a list into a local / remote .txt file.

    Each item of the list will be saved on a new line of the .txt file.

    Args:
        data: List of items to save to .txt file.
        file_path: Local or remote .txt file to save items of `data`.
        force: Indicates if the file should be overwritten if it already exists.
        use_background_thread: Whether to save the file with a background thread.
            If set to True, the `_save_file` function will be called again in a separated thread.

    Raises:
        FileExistsError: The file already exists and force=False.
    """

    def _save_items(data: list, file_obj: TextIO) -> None:
        for item in data:
            file_obj.write(f"{item}\n")

    _save_file(
        obj=data,
        file_path=file_path,
        save_fn=_save_items,
        force=force,
        mode="w",
        use_background_thread=use_background_thread,
    )


def load_yml(file_path: Openable, **kwargs: Any) -> dict:
    """Load a local / remote .yml file.

    Args:
        file_path: Local or remote .yml file to load.
        kwargs: Keyword arguments to forward to `yaml.full_load`.

    Raises:
        FileNotFoundError: The 'file_path' provided doesn't exist.
    """
    return _load_file(file_path=file_path, load_fn=yaml.full_load, mode="r", **kwargs)


def save_yml(
    dict_: dict,
    file_path: Openable,
    force: bool = False,
    use_background_thread: bool = False,
    **kwargs: Any,
) -> None:
    """Save a dict into a local / remote .yml file.

    Remarks:
    - by default:
        - `default_flow_style=False`
        - `sort_keys=False`
    - all keyword arguments will be forwarded to `yaml.dump` function

    Args:
        dict_: Python dictionary to save to .yml file.
        file_path: Local or remote .yml file to save `dict_`.
        force: Indicates if the file should be overwritten if it already exists.
        use_background_thread: Whether to save the file with a background thread.
            If set to True, the `_save_file` function will be called again in a separated thread.
        kwargs: Keyword arguments to forward to `yaml.dump`.

    Raises:
        FileExistsError: The file already exists and force=False.
    """
    kwargs["default_flow_style"] = kwargs.get("default_flow_style", False)
    kwargs["sort_keys"] = kwargs.get("sort_keys", False)
    _save_file(
        obj=dict_,
        file_path=file_path,
        save_fn=yaml.dump,
        force=force,
        mode="w",
        use_background_thread=use_background_thread,
        **kwargs,
    )


def load_pkl(
    file_path: Openable,
    unpickler_cls: type[pickle.Unpickler] = pickle.Unpickler,
    **kwargs: Any,
) -> Any:
    """Load a local / remote .pkl file.

    Args:
        file_path: Local or remote .pkl to load.
        unpickler_cls: Class used to unpickle the file. By default, the `pickle.Unpickler` is used.
        kwargs: Keyword arguments to forward to `unpickler_cls.__init__`.

    Returns:
        Python object deserialized.

    Raises:
        FileNotFoundError: The 'file_path' provided doesn't exist.
    """
    return _load_file(
        file_path=file_path,
        load_fn=lambda f, **kwargs: unpickler_cls(f, **kwargs).load(),
        mode="rb",
        **kwargs,
    )


def save_pkl(
    obj: Any,
    file_path: Openable,
    force: bool = False,
    use_background_thread: bool = False,
    **kwargs: Any,
) -> None:
    """Save a Python object into a local / remote .pkl file thanks to pickle package.

    Args:
        obj: Python object to save as in .pkl file.
        file_path: Local or remote .pkl file to save `obj`.
        force: Indicates if the file should be overwritten if it already exists.
        use_background_thread: Whether to save the file with a background thread.
            If set to True, the `_save_file` function will be called again in a separated thread.
        kwargs: Keyword arguments to forward to `pickle.dump`.

    Raises:
        FileExistsError: The file already exists and force=False.
    """
    _save_file(
        obj=obj,
        file_path=file_path,
        save_fn=pickle.dump,
        force=force,
        mode="wb",
        use_background_thread=use_background_thread,
        **kwargs,
    )


def load_npy(file_path: Openable, **kwargs: Any) -> np.ndarray:
    """Load a local / remote .npy file.

    Args:
        file_path: Local or remote .npy to load.
        kwargs: Keyword arguments to forward to `np.load`.

    Returns:
        Numpy array stored in the file.
    """
    return _load_file(file_path=file_path, load_fn=np.load, from_file_obj=False, **kwargs)


def save_npy(
    array: np.ndarray,
    file_path: Openable,
    force: bool = False,
    use_background_thread: bool = False,
    **kwargs: Any,
) -> None:
    """Save a numpy array into a local / remote .npy file.

    Args:
        array: A numpy array to save in .npy file.
        file_path: Local or remote .npy file to save `array`.
        force: Indicates if the file should be overwritten if it already exists.
        use_background_thread: Whether to save the file with a background thread.
            If set to True, the `_save_file` function will be called again in a separated thread.
        kwargs: Keyword arguments to forward to `np.save`.

    Raises:
        FileExistsError: The file already exists and force=False.
    """
    _save_file(
        obj=array,
        file_path=file_path,
        # np.save has a signature different from other save functions
        # we must use a lambda function to match the other signatures
        save_fn=lambda array, file, **kwargs: np.save(file, array, **kwargs),
        force=force,
        mode="wb",
        use_background_thread=use_background_thread,
        **kwargs,
    )


def load_json(file_path: Openable, **kwargs: Any) -> dict:
    """Load a local / remote .json file.

    Args:
        file_path: Local or remote .json to load.
        kwargs: Keyword arguments to forward to `json.load`.

    Returns:
        Python dictionary.

    Raises:
        FileNotFoundError: The 'file_path' provided doesn't exist.
    """
    return _load_file(file_path=file_path, load_fn=json.load, mode="r", **kwargs)


def save_json(
    data: dict,
    file_path: Openable,
    force: bool = False,
    use_background_thread: bool = False,
    **kwargs: Any,
) -> None:
    """Save a dict into a local / remote .json file.

    Remarks:
    - by default, 'indent=4' is used for a better display.
    - all keyword arguments will be forwarded to 'json.dump' function

    Args:
        data: Python dictionary to save as in .json file.
        file_path: Local or remote .json file to save `data`.
        force: Indicates if the file should be overwritten if it already exists.
        use_background_thread: Whether to save the file with a background thread.
            If set to True, the `_save_file` function will be called again in a separated thread.
        kwargs: Keyword arguments to forward to `json.dump`.

    Raises:
        FileExistsError: The file already exists and force=False.
    """
    kwargs["indent"] = kwargs.get("indent", 4)
    _save_file(
        obj=data,
        file_path=file_path,
        save_fn=json.dump,
        force=force,
        mode="w",
        use_background_thread=use_background_thread,
        **kwargs,
    )


def load_csv(file_path: Openable, **kwargs: Any) -> pd.DataFrame:
    """Load a local / remote .csv file with pandas.

    Args:
        file_path: Local or remote .csv to load.
        kwargs: Keyword arguments to forward to `pd.read_csv`.

    Returns:
        A pandas.DataFrame loaded from CSV file.

    Raises:
        FileNotFoundError: The 'file_path' provided doesn't exist.
        ValueError: There is a mismatch between the 'compression' provided and the file's extension.
    """
    # default value in pd.DataFrame.to_csv
    # cf https://pandas.pydata.org/pandas-docs/version/1.4/reference/api/pandas.read_csv.html
    compression = kwargs.get("compression", "infer")

    if not _is_valid_compression(file_path, compression):
        raise ValueError(
            f"Mismatch between 'compression={compression}' and file's extension {file_path}"
        )

    return _load_file(file_path=file_path, load_fn=pd.read_csv, from_file_obj=False, **kwargs)


def save_csv(
    df: pd.DataFrame,
    file_path: Openable,
    force: bool = False,
    mode: str = "w",
    use_background_thread: bool = False,
    **kwargs: Any,
) -> None:
    """Save a pandas dataframe to a local / remote file.

    Remarks:
    - all keyword arguments will be forwarded to `pd.DataFrame.to_csv` function

    Args:
        df: A pandas.DataFrame to save.
        file_path: Path to save the pandas.DataFrame.
        force: Indicates if the file should be erased if it already exists.
        mode: Mode to open the file.
        use_background_thread: Whether to save the file with a background thread.
        kwargs: Keyword arguments to forward to `pd.DataFrame.to_csv`

    Raises:
        FileExistsError: The file already exists and force=False.
        ValueError: There is a mismatch between the 'compression' provided and the file's extension.
    """
    # default value in pd.DataFrame.to_csv
    # cf https://pandas.pydata.org/pandas-docs/version/1.4/reference/api/pandas.DataFrame.to_csv.html # noqa
    compression = kwargs.get("compression", "infer")

    if not _is_valid_compression(file_path, compression):
        raise ValueError(
            f"Mismatch between 'compression={compression}' and file's extension {file_path}"
        )

    _save_file(
        obj=df,
        file_path=file_path,
        save_fn=pd.DataFrame.to_csv,
        force=force,
        mode=mode,
        use_background_thread=use_background_thread,
        from_file_obj=False,
        **kwargs,
    )


def load_fasta(file_path: Openable, **kwargs: Any) -> pd.DataFrame:
    """Load a local or remote fasta file.

    Args:
        file_path: Local or remote .fasta to load.
        kwargs: Keyword arguments to forward to `_process_fasta_file`.

    Returns:
        A pandas.DataFrame loaded from .fasta file.

    Raises:
        FileNotFoundError: The 'file_path' provided doesn't exist.
    """
    return _load_file(file_path, load_fn=_process_fasta_file, mode="r", **kwargs)


def _save_file(
    obj: Any,
    file_path: str | os.PathLike,
    save_fn: Callable,
    force: bool,
    mode: str,
    use_background_thread: bool,
    from_file_obj: bool = True,
    **save_fn_kwargs: Any,
) -> None:
    """Main interface to save a file locally or remotely.

    Support for remote files is done via cloudpathlib.

    When saving a file remotely, it will first create a local copy in the
    cloudpathlib's cache and then upload it.

    The local cache used is defined by:
    - the env CLOUDPATHLIB_LOCAL_CACHE_DIR if set.
    - otherwise, a random /tmp subdirectory.

    Args:
        obj: Python object to save.
        file_path: Path to the local or remote file.
        save_fn: Function used to save the file.
        force: Indicates if the file should be overwritten if it already exists.
        mode: mode to use to open the file.
        use_background_thread: whether to save the file with a background thread.
            If set to True, the `_save_file` function will be called again in a separated thread.
        from_file_obj: indicates if the `save_fn` should be used on the path as string or on the
            file object.
        save_fn_kwargs: Keyword arguments to forward to `save_fn`.

    Raises:
        FileExistsError: The file already exists and force=False.
    """
    if use_background_thread:
        threading.Thread(
            target=_save_file,
            kwargs={
                "obj": obj,
                "file_path": file_path,
                "save_fn": save_fn,
                "force": force,
                "mode": mode,
                "use_background_thread": False,
                "from_file_obj": from_file_obj,
                **save_fn_kwargs,
            },
            name=f"save_{file_path}",
        ).start()

        return

    # In case of append mode, we should remove the check
    if mode == "a":
        force = True

    file_path_ = anypath.to_anypath(file_path)

    if not force and file_path_.exists():
        raise FileExistsError(
            f"The file {file_path} already exist, set force=True if you want to overwrite it."
        )

    file_path_.parent.mkdir(exist_ok=True, parents=True)

    # Make sure to create local directory in the cache
    # It is useful in case 'from_file_obj=False' since we are calling 'save_fn' on
    # the output os.fspath == path to local cache
    if isinstance(file_path_, CloudPath):
        file_path_._local.parent.mkdir(exist_ok=True, parents=True)

    if from_file_obj:
        with file_path_.open(mode) as f:
            save_fn(obj, f, **save_fn_kwargs)
    else:
        # Useful for pd.DataFrame.to_csv which supports 'mode' as parameter.
        if _should_add_mode_to_kwargs(save_fn):
            save_fn_kwargs["mode"] = mode

        # 'os.fspath' corresponds to:
        # - for pathlib.Path: the path as str
        # - for cloudpathlib.CloudPath: the path as str to the local cache
        save_fn(obj, os.fspath(file_path_), **save_fn_kwargs)

        # For CloudPath, we have to manually upload the local file
        if isinstance(file_path_, CloudPath):
            file_path_._upload_local_to_cloud(force_overwrite_to_cloud=force)


def _load_file(
    file_path: Openable,
    load_fn: Callable,
    mode: str | None = None,
    from_file_obj: bool = True,
    **load_fn_kwargs: Any,
) -> Any:
    """Main interface to load a local or remote file.

    Support for remote files is done via cloudpathlib.

    A local cache is always used:
    - if env CLOUDPATHLIB_LOCAL_CACHE_DIR is set, use it.
    - otherwise a random /tmp directory is used during the full process.

    Args:
        file_path: Path to the local or remote file.
        load_fn: Function used to load the file.
        mode: mode to use to open the file.
        from_file_obj: indicates if the 'load_fn' should be used on the path as string or on the
            file object. If set to False, the 'mode' parameter is ignored.
        load_fn_kwargs: Keyword arguments to forward to load_fn.

    Returns:
        Loaded file, the type depends on the 'load_fn' provided.

    Raises:
        ValueError: If 'from_file_obj=True' and 'mode' is not provided.
        FileNotFoundError: The 'file_path' provided doesn't exist.
    """
    if from_file_obj and mode is None:
        raise ValueError("If 'from_file_obj=True', you must provide 'mode' parameter.")

    file_path_ = anypath.to_anypath(file_path)

    if not file_path_.exists():
        raise FileNotFoundError(f"The file {file_path} can't be found.")

    if from_file_obj:
        with file_path_.open(mode) as f:
            obj = load_fn(f, **load_fn_kwargs)
    else:
        # For CloudPath, 'fspath' will download the file locally first
        obj = load_fn(os.fspath(file_path_), **load_fn_kwargs)

    return obj


def _should_add_mode_to_kwargs(save_fn: Callable) -> bool:
    """Indicates if 'mode' is a valid argument for the `save_fn`."""
    return "mode" in inspect.signature(save_fn).parameters


def _is_valid_compression(file_path: Openable, compression: str | None) -> bool:
    """Check if there is no mismatch between file_path's extension and compression.

    Raises:
        ValueError: The 'compression' provided is not supported.
    """
    if compression == "infer":
        return True

    file_path = Path(file_path)

    # no compression: file_path must have only one suffix (e.g. .csv)
    if compression is None:
        return len(file_path.suffixes) == 1

    try:
        return COMPRESSION2EXTENSION[compression] == file_path.suffix
    except KeyError:
        raise ValueError(
            f"compression={compression} is not supported. Use one of the following "
            f"{list(COMPRESSION2EXTENSION.keys())}."
        )


def _process_fasta_file(file: TextIO, columns: list[str]) -> pd.DataFrame:
    """Process fasta file to be loaded to pandas.DataFrame."""
    key2seq: DefaultDict[str, str] = defaultdict(str)

    for line in file:
        if line[0] == ">":
            newkey = line[1:-1]
            continue

        key2seq[newkey] += line.strip()

    return pd.DataFrame(key2seq.items(), columns=columns)
