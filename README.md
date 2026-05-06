# ML Final Project

## Local Setup

1. Clone the repo and `cd` into it.

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the N-CMAPSS DS02 dataset from Kaggle and drop `N-CMAPSS_DS02-006.h5` into `data/raw/`:

5. Run the notebooks in order from `notebooks/`:
   - `00_sanity_check.ipynb`
   - `01_eda.ipynb`
   - `02_features.ipynb`
   - `03_Modelling.ipynb`

## Tests

```bash
pytest
```
