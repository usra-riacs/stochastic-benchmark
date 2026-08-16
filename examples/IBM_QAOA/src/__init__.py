# Present so this "src" package (examples/IBM_QAOA/src) is a regular
# package rather than merging with the core repo's src/ (pyproject.toml
# sets pythonpath = "src") into one ambiguous namespace package when both
# are on sys.path at once, e.g. in tests/test_ibm_qaoa_processing.py.
