import json
import subprocess
import sys

from hybrid_retrieval import __version__, scaffold_info


def test_scaffold_metadata_and_smoke_cli():
    info = scaffold_info()
    assert info["version"] == __version__
    assert info["pipeline"] == ("bm25", "dense", "rrf", "crossencoder")
    proc = subprocess.run(
        [sys.executable, "-m", "hybrid_retrieval.smoke"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = json.loads(proc.stdout)
    output["pipeline"] = tuple(output["pipeline"])
    assert output == info
