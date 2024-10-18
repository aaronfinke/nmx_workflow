from pathlib import Path
from nmx_workflow.config import META_DATA_DIR

def file_hash(filepath: Path) -> str:
    # Simplified from Pooch -- we only care about sha256 for now
    import hashlib
    import functools
    chunk_size = 65536
    hasher = getattr(hashlib, 'sha256', functools.partial(hashlib.new, 'sha256'))()
    with filepath.open('rb') as filein:
        chunk = filein.read(chunk_size)
        while chunk:
            hasher.update(chunk)
            chunk = filein.read(chunk_size)
    return hasher.hexdigest()


def make_registry(base, dirs, recursive=True, ext=None):
    pat = '**/*' if recursive else '*'
    if ext is not None:
        pat = f'{pat}{ext}'
    hashes = {p.relative_to(base).as_posix(): file_hash(p) for d in dirs for p in base.joinpath(d).glob(pat) if
              p.is_file()}
    return hashes


def make_nextcloud_urler(base_url: str, folder_id: str, root: str):
    """For a Nextcloud base URL, and the id of a shared folder, make a function to construct the full URL for a file

    Parameters
    ----------
    base_url: str
        The url at which the nextcloud instance is running, e.g. "https://project.esss.dk/nextcloud"
    folder_id: str
        The (valid) shared folder identifier, e.g. 'Diq9n3kITaEBtq7'

    Returns
    -------
    The full download path, e.g., for the above input
    "https://project.esss.dk/nextcloud/index.php/s/Diq9n3kITaEBtq7/download?path=%2FSim&files=BIFROST.h5",
    """
    from urllib.parse import quote_plus
    url = (base_url if base_url.endswith('/') else base_url + '/') + f'index.php/s/{quote_plus(folder_id)}'

    def nextcloud_url(name: str):
        f"""Return the Nextcloud download URL for a file located under {url}"""
        path_file = Path(root) / name
        return f'{url}/download?path={quote_plus(str(path_file.parent))}&files={str(path_file.name)}'

    return nextcloud_url


def write_registry(hashes: dict[str, str], output: Path, urler):
    lines = '\n'.join(f'{name} {hashes[name]} {urler(name)}' for name in sorted(hashes.keys()))
    with output.open('w') as file:
        file.write(lines + '\n')


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser("Nextcloud pooch registrar")
    parser.add_argument('--root', type=Path, help='register relative to root path')
    parser.add_argument('-d', '--dirs', type=str, default='',
                        help='comma separated list of directories relative to root to include')
    parser.add_argument('-r', '--recursive', action='store_true', help='recurse through list of directories')
    parser.add_argument('--no-recursive', action='store_false', dest='recursive', help='disable recursion')
    parser.add_argument('--ext', type=str, default='', help='limit search to only this extension')
    parser.add_argument('-o', '--output', type=str, default='pooch-registry.txt', help='registry file name')
    parser.add_argument('-n', type=str, help='Nextcloud base url', default='https://project.esss.dk/nextcloud')
    parser.add_argument('-s', type=str, help='Nextcloud shared folder', default='Db983rx3b67wxbR')
    args = parser.parse_args()

    root = Path(args.root) if isinstance(args.root, str) else args.root
    dirs = [''] if args.dirs == '' else args.dirs.split(',')
    ext = None if args.ext == '' else args.ext
    hashes = make_registry(root, dirs, recursive=args.recursive, ext=ext)
    if len(hashes):
        urler = make_nextcloud_urler(args.n, args.s, root.name)
        write_registry(hashes, META_DATA_DIR / args.output, urler)


if __name__ == '__main__':
    main()