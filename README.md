# Test Project

A Python application with parser, analyzer, dashboard, and agent components.

## Installation

```bash
pip install -e .
```

## Usage

This project provides the following modules:

- `app.parser`: Data parsing functionality
- `app.analyzer`: Data analysis functionality
- `app.dashboard`: Dashboard interface
- `app.agent`: Agent operations

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
black --check .
ruff check .
```

## Project Structure

```
.
├── README.md
├── pyproject.toml
├── app/
│   ├── __init__.py
│   ├── parser.py
│   ├── analyzer.py
│   ├── dashboard.py
│   └── agent.py
└── tests/
    ├── __init__.py
    ├── test_parser.py
    ├── test_analyzer.py
    ├── test_dashboard.py
    └── test_agent.py
```

## Requirements

- Python 3.8+
- Standard library only
- pytest for testing

## License

MIT
