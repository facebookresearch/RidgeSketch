# Development Workflow

1. `git checkout -b feature-name`
2. commit work as usual to your branch
3. `git push origin feature-name`
4. create a pull request for merging into master
   - others can review / comment on code before merging


Code formatting uses [Black](https://black.readthedocs.io/en/stable/).
To run manually: `black .` from the parent directory.
- This should automatically run as a pre-commit hook

To store requirements: `pip freeze --exclude-editable > requirements.txt`

# Run Tests
[PyTest](https://docs.pytest.org/en/latest/) is our testing framekwork.

Run all test: `pytest ` from the parent directory.

Test coverage report: `pytest --cov-config=.coveragerc --cov=. --runslow`

# Generate Documentation

Install Sphinx `pip install sphinx`. From the `docs` directory,

1. `sphinx-build -b html . _build`
2. `make html`

To view docs: `open _build/index.html`

To edit the contents change `index.rst`
