"""Helper for tests/test_inverse_results.py: run one inverse-design query in a fresh process, print its rows as csv
(written like the result files, so float32 values keep their float32 repr).

usage: python tests/_inverse_query.py METHOD TRAIN SEED GEOM_ID      (PYTHONPATH=src; thread limits from the caller)
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    warnings.filterwarnings("ignore")
    from cabd import data
    from cabd.inverse.run import one_query
    method, train, seed, geom = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    q = data.queries("final").set_index("geom_id", drop=False)
    rows = one_query(method, train, "final", seed, 20, q.loc[geom])
    import pandas as pd
    print("ROWS")
    print(pd.DataFrame(rows).to_csv(index=False), end="")


if __name__ == "__main__":
    main()
