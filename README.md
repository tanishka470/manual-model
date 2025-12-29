# Manual Marks Predictor (Hours → Marks)

A simple, manual linear regression model that learns the relationship between hours studied and marks using gradient descent. It prints training loss, learned parameters (`w`, `b`), shows a scatter + best-fit line, and prompts you to enter hours to predict marks.

## Features
- Gradient descent training on sample data
- Mean Squared Error (MSE) loss reporting
- Plot of data and fitted line
- Interactive prediction: prompt for hours, output predicted marks

## Quick Start

### 1) Install Python dependencies
Using pip:

```powershell
pip install -r requirements.txt
```

Or individually:

```powershell
pip install numpy
pip install matplotlib
```

### 2) Run the script
From this folder:

```powershell
python student.py
```

You will see training logs and be prompted:
```
Enter hours studied: 6
```
Then you’ll get the predicted marks and a plot window.

## How It Works
- Model: `y_pred = w * x + b`
- Loss (MSE): average of `(y - y_pred)^2`
- Updates per epoch:
  - `dw = (-2/n) * sum(x * (y - y_pred))`
  - `db = (-2/n) * sum(y - y_pred)`
- Parameters updated with learning rate `lr`:
  - `w = w - lr * dw`
  - `b = b - lr * db`

See the implementation in `student.py`.

## Customize Data
The sample arrays are defined in `student.py`:
- `x = np.array([1,2,3,4,5])`  (hours)
- `y = np.array([20,30,40,50,60])` (marks)

You can replace them with your own values. The file `data.csv` is currently unused; if you want, we can wire the script to load hours/marks from CSV.

## Troubleshooting
- If the prompt doesn’t appear until after plotting, we’ve moved input before `plt.show()` to ensure it asks first.
- If `matplotlib` fails to show a window, ensure it’s installed and you’re not running in a headless environment.

## Next Steps
- Add CSV loading support
- Log additional metrics (MAE, R²)
- Save the fitted `w` and `b` for reuse
