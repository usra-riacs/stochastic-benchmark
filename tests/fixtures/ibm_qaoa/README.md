# IBM QAOA Processing Test Fixtures

This directory contains test fixtures for validating the IBM QAOA data ingestion pipeline in `examples/IBM_QAOA/src/Processing.py`.

## Real Data Fixtures

The following files are real IBM QAOA experimental results used for integration testing:

- **`20250901_165018_000N10R3R_MC_FA_SV_noOpt_2.json`** - Instance 000, depth 2, single trial
- **`20250901_165018_000N10R3R_MC_FA_SV_noOpt_4.json`** - Instance 000, depth 4, single trial
- **`20250913_170712_001N10R3R_MC_FA_SV_noOpt_1.json`** - Instance 001, depth 1, single trial
- **`20250913_171721_002N10R3R_MC_FA_SV_noOpt_1.json`** - Instance 002, depth 1, single trial

### Filename Format
IBM QAOA files follow the pattern:
```
YYYYMMDD_HHMMSS_###N##R3R_MC_FA_SV_noOpt_#.json
```
Where:
- `YYYYMMDD_HHMMSS` - Timestamp
- `###` - Instance ID (e.g., 000, 001, 002)
- `N##R#R` - N## = Problem Size, R#R = # Random Regular Graph
- `_#.json` - QAOA parameter (p)

## Synthetic Fixtures

### `multi_trial_synthetic.json`
A synthetic file with 3 trials to test multi-trial processing and standard bootstrap behavior.

**Schema:**
```json
{
  "0": {
    "energy": -12.5,
    "approximation ratio": 0.85,
    "train_duration": 2.3,
    "trainer": {
      "trainer_name": "FixedAngleConjecture",
      "evaluator": null
    },
    "success": true,
    "optimal_params": [0.1, 0.2],
    "history": [-10.0, -11.5, -12.5]
  }
}
```

### `missing_trainer.json`
Edge case: Trial without `trainer` field (tests default "Unknown" trainer assignment).

### `missing_optimal_params.json`
Edge case: Trial without `optimal_params` field (tests empty parameter handling).

### `empty_trials.json`
Edge case: JSON with no numeric trial keys (tests empty data handling).

## JSON Schema

Standard IBM QAOA trial structure:
```json
{
  "<trial_id>": {
    "energy": <float>,
    "approximation ratio": <float>,
    "train_duration": <float>,
    "trainer": {
      "trainer_name": <string>,
      "evaluator": <string|null>
    },
    "success": <boolean>,
    "optimal_params": [<float>, ...],
    "history": [<float>, ...]
  }
}
```

**Required Fields:** None (all have defaults in parsing logic)

**Default Values:**
- `energy`, `approximation ratio`: `NaN`
- `train_duration`: `0.0`
- `trainer_name`: `"Unknown"`
- `evaluator`: `None`
- `success`: `False`
- `optimal_params`: `[]`
- `history`: `[]`

## Test Coverage

The test suite (`tests/test_ibm_qaoa_processing.py`) validates:

### Unit Tests
- `QAOAResult` dataclass creation
- `parse_qaoa_trial`: field extraction, defaults, trainer variations
- `load_qaoa_results`: single/multi-trial, numeric key filtering
- `convert_to_dataframe`: column mapping, parameter extraction
- `group_name_fcn`: filename parsing
- `prepare_stochastic_benchmark_data`: pickle serialization

### Integration Tests
- Single/multiple file processing
- GTMinEnergy proxy injection
- Pickle file creation
- Single-trial bootstrap fabrication
- Interpolation fallback for minimal data
- Train/test split addition
- Empty pattern handling

### Edge Cases
- Missing trainer field
- Missing optimal_params field
- Empty trials dictionary
- Multi-trial synthetic data

## Usage

Tests automatically discover fixtures via:
```python
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "ibm_qaoa"
```

Run tests:
```bash
pytest tests/test_ibm_qaoa_processing.py -v
```

Run with coverage:
```bash
pytest tests/test_ibm_qaoa_processing.py --cov=examples/IBM_QAOA --cov-report=term-missing
```

## Maintenance

When adding new fixtures:
1. Place JSON files in this directory
2. Follow IBM naming convention for real data
3. Use descriptive names for synthetic edge cases
4. Update this README with fixture description
5. Add corresponding test case in `test_ibm_qaoa_processing.py`

## Boundaries

**IBM-Specific (stays in `examples/IBM_QAOA/`):**
- Filename parsing regex (`###N##R3R_..._#.json`)
- JSON schema assumptions
- Single-trial bootstrap fabrication logic
- GTMinEnergy proxy insertion
- Interpolation skip heuristic (≤5 rows)

**Generic (in `src/`):**
- Bootstrap framework (`bootstrap.py`)
- Interpolation utilities (`interpolate.py`)
- Success metrics (`success_metrics.py`)
- Stochastic benchmark orchestration (`stochastic_benchmark.py`)
