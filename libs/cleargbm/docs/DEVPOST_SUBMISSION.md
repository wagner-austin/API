# ClearGBM — Interpretable Gradient Boosting

## Inspiration

I wanted to understand how gradient boosting actually works, not the API, but the math and mechanics underneath.

## What it does

ClearGBM is a gradient boosting machine built from scratch in pure Python with zero dependencies. Every prediction includes the exact rules and feature contributions that produced it.

## How I built it

Started with binary log loss and its derivatives. Built decision trees that use gradients to find optimal splits. Added the boosting loop where each tree corrects the previous. Implemented histogram binning to bucket values into 64 bins for faster splits. Added sibling subtraction to derive child histograms from the parent. Built rule extraction and feature contribution tracking on top.

## Challenges I ran into

Zero dependencies meant no numpy arrays, so everything is tuples and pure stdlib. No sklearn metrics, instead I wrote my own AUC. Every piece had to be understood and implemented, not imported.

## Accomplishments that I'm proud of

Tested on the US Bankruptcy dataset (78,682 samples) with Optuna hyperparameter optimization:

| Rank | Backend  | Features | AUC    | Time     |
|------|----------|----------|--------|----------|
| 1st  | LightGBM | 495      | 0.8816 | 86s      |
| 2nd  | LightGBM | 324      | 0.8754 | 85s      |
| 3rd  | ClearGBM | 495      | 0.8737 | 7327s    |
| 4th  | ClearGBM | 324      | 0.8704 | 5400s    |
| 5th  | XGBoost  | 495      | 0.8491 | 179s     |

ClearGBM beats XGBoost and comes within 1% of LightGBM — with zero external dependencies and built-in interpretability. Strict mypy typing with 100% test coverage.

## What I learned

How gradient and hessian sums determine optimal splits. Why histogram binning reduces complexity from O(n log n) to O(k). How L1 regularization soft-thresholds leaf values. The mechanics behind every "magic" library call.

## What's next for ClearGBM

Adding numpy as an optional backend would dramatically improve speed — the current 7000s could drop to under 100s — while keeping the pure Python fallback for environments without numpy. The accuracy is already there; speed is just an implementation detail. Also planned: multi-class classification, regression support, and GPU acceleration.

## Built With

Python
