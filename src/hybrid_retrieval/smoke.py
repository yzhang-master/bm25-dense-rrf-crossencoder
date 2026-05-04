import json

from . import scaffold_info


def main() -> None:
    info = scaffold_info()
    if len(info["pipeline"]) != 4:
        raise RuntimeError("invalid retrieval pipeline metadata")
    print(json.dumps(info, sort_keys=True))


if __name__ == "__main__":
    main()
