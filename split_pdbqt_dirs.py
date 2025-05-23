#!/usr/bin/env python3
"""
fast_split_pdbqt_sendfile.py ▸ Ultra-fast splitter using os.rename & os.sendfile.

Features
--------
• Batches of up to CHUNK files (default 50 000).
• Optional shuffling (--random).
• Moves by default; copies use zero-copy os.sendfile.
• Multiprocessing pool (default = cpu_count()).
• No early returns—safe in REPL/Jupyter.
"""

import os, sys, argparse, random
from pathlib import Path
from multiprocessing import Pool, cpu_count

def copy_sendfile(src: Path, dst: Path):
    st = src.stat()
    with src.open('rb') as fsrc, dst.open('wb') as fdst:
        offset = 0
        while offset < st.st_size:
            sent = os.sendfile(fdst.fileno(), fsrc.fileno(), offset, st.st_size - offset)
            if sent == 0:
                break
            offset += sent
    os.utime(dst, (st.st_atime, st.st_mtime))

def move_or_copy(src: Path, dst_dir: Path, do_copy: bool) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    try:
        if do_copy:
            # kernel-accelerated copy
            copy_sendfile(src, dst)
        else:
            os.rename(src, dst)
    except OSError:
        # fallback for cross-fs moves
        copy_sendfile(src, dst)
        try:
            src.unlink()
        except OSError:
            pass

def process_batch(args):
    idx, file_list, dst_root, prefix, do_copy = args
    dest = dst_root / f"{prefix}{idx:04d}"
    count = 0
    for src in file_list:
        move_or_copy(src, dest, do_copy)
        count += 1
    return count

def main():
    p = argparse.ArgumentParser(description="Split huge PDBQT dir lightning-fast.")
    p.add_argument("--src",       required=True, type=Path, help="flat dir of .pdbqt files")
    p.add_argument("--dst",       required=True, type=Path, help="output root directory")
    p.add_argument("--chunk",     type=int, default=50_000, help="max files per batch")
    p.add_argument("--processes", type=int, default=cpu_count(), help="number of worker processes")
    p.add_argument("--ext",       default="pdbqt", help="file extension (no dot)")
    p.add_argument("--prefix",    default="batch_", help="sub-dir name prefix")
    p.add_argument("--copy",      action="store_true", help="copy instead of move")
    p.add_argument("--random",    action="store_true", help="shuffle files before splitting")
    args = p.parse_args()

    if not args.src.is_dir():
        sys.exit(f"[ERROR] Source {args.src} not found or not a directory.")
    args.dst.mkdir(parents=True, exist_ok=True)

    # 1) Gather all filenames
    files = []
    with os.scandir(args.src) as it:
        for ent in it:
            if ent.is_file() and ent.name.lower().endswith(f".{args.ext.lower()}"):
                files.append(Path(ent.path))
    total = len(files)
    if total == 0:
        sys.exit("[INFO] No files matched; nothing to do.")

    # 2) Shuffle if requested
    if args.random:
        random.shuffle(files)

    # 3) Partition into exact chunks
    batches = [files[i : i + args.chunk] for i in range(0, total, args.chunk)]
    n_batches = len(batches)
    print(f"[INFO] {total:,} files → {n_batches:,} batches of ≤{args.chunk:,}.")

    # 4) Build pool tasks
    tasks = [
        (idx, batch, args.dst, args.prefix, args.copy)
        for idx, batch in enumerate(batches)
    ]
    workers = min(args.processes, n_batches)
    print(f"[INFO] Launching {workers} worker processes…")

    # 5) Run
    moved = 0
    with Pool(workers) as pool:
        for cnt in pool.imap_unordered(process_batch, tasks):
            moved += cnt
            if moved % 10000 == 0 or moved == total:
                print(f"[INFO] {moved:,}/{total:,} files done…")

    print(f"[DONE] Processed {moved:,} files into {n_batches:,} folders.")

if __name__ == "__main__":
    main()
