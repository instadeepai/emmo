# Remote files support

- All the functions defined in `emmo/io/file.py` allow to load and save files locally and remotely
  on GCP: you can find
  [here](<https://console.cloud.google.com/storage/browser?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&project=biontech-tcr&prefix=&forceOnObjectsSortingFiltering=false&forceOnBucketsSortingFiltering=true>)
  the list of buckets available.
- It is done thanks to [cloudpathlib](https://cloudpathlib.drivendata.org/stable/) which allows to
  use `pathlib`-like syntax for remote files via `cloudpathlib.CloudPath`.
- If a variable can correspond to a local or a remote file you can use `cloudpathlib.AnyPath` class
  which will take care of creating either a `pathlib.Path` or a `cloudpathlib.CloudPath` instance.
- `cloudpathlib` is using a local cache to avoid to download several times the same file. It can be
  configured thanks to env variable `CLOUDPATHLIB_LOCAL_CACHE_DIR` (set by default in
  `.env.template`).
