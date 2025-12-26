# ClearGBM Demo Video Script (~5 minutes)

---

## INTRO (45 seconds)

"I built a gradient boosting machine from scratch called ClearGBM.

I'm running it through a testing harness and optimization script right now. The harness uses Optuna, an optimizer that tries different combinations of settings — like the learning rate, which controls how much each tree corrects the previous one, and n_estimators, which is just how many trees to build — and finds the best combination.

This lets me compare ClearGBM directly against XGBoost and LightGBM, the industry standard libraries.

The difference is I wrote ClearGBM using only Python's standard library. Not even numpy — which would make the math faster but wouldn't improve accuracy. So the accuracy we're getting is purely from the algorithm design, not from optimized C code."

---

## WHAT THESE MODELS DO (45 seconds)

"The goal of these models is to learn patterns from giant tables of data — like Excel spreadsheets with thousands of rows.

For this demo, I'm using US bankruptcy data. Each row is a consumer with various financial data points — debt ratios, income, credit history. The column headers are the features. The model learns which features matter most when predicting whether someone will default on their credit.

By the end, the model tells you: these are the most important columns, and here are the exact rules for how they combine to predict default risk."

---

## HOW CLEARGBM WORKS (90 seconds)

"Let me explain exactly what ClearGBM does, step by step.

First, it looks at all the training data and makes an initial guess — just the average default rate. If 6% of people in the data defaulted, it starts by predicting 6% for everyone.

Then it builds a decision tree. But not by looking at accuracy directly. It calculates two things for each data point: the gradient, which is how wrong the current prediction is, and the hessian, which is how confident we should be in correcting it. The tree finds splits that group similar gradients together.

Here's where my optimization comes in. Instead of sorting all 78,000 samples to find the best split point — which is slow — I bucket each feature's values into 64 bins. Then I just scan 64 numbers instead of sorting 78,000. Same result, much faster. That's histogram binning.

After the first tree makes predictions, the second tree focuses only on the errors the first tree made. The third tree focuses on the remaining errors. Each tree gets multiplied by the learning rate — usually 0.1 — so it only makes small corrections. After 100 trees, you combine all their outputs.

I also added L1 and L2 regularization. L1 pushes weak leaf values toward zero — if a split barely helps, ignore it. L2 shrinks all leaf values to prevent any single tree from overcorrecting. This stops the model from memorizing noise in the training data."

---

## THE RESULTS (60 seconds)

*Show terminal with optimization results*

"Here's the final comparison. I ran all three models through Optuna optimization — 5 trials each, testing different learning rates and tree counts, across different feature engineering presets.

On the US bankruptcy dataset — 78,000 consumers — here are the results:

LightGBM: 0.8816 AUC
ClearGBM: 0.8737 AUC
XGBoost: 0.8491 AUC

ClearGBM beats XGBoost by a significant margin — 0.8737 versus 0.8491. And it's within 1% of LightGBM.

That's a from-scratch Python implementation, no C++ libraries, competing with industry standard tools.

The tradeoff is speed — ClearGBM took about 2 hours while LightGBM took 86 seconds. That's because every loop, every array operation is pure Python. Adding numpy as an optional backend would cut that down dramatically — probably under 100 seconds — because numpy does array math in C behind the scenes.

But that's just an implementation detail. The accuracy proves the algorithm design is correct. Speed is fixable; getting the math right is the hard part."

---

## INTERPRETABILITY (60 seconds)

*Show rule extraction output*

"But here's what makes ClearGBM different. For every prediction, I can extract exactly why.

I trace the path each data point takes through all 100 trees. Then I convert those paths into readable rules:

'If X6 is greater than negative 2.75, and X8 is greater than 714, and X1 is greater than 155 — then low default risk.'

That's not a black box. That's a rule you can read, audit, and explain to a regulator.

I also track feature contributions — exactly how many points each column added or subtracted from the final prediction. X8 contributed 14% of the prediction power. X15 contributed 10%. You can see which columns matter most.

XGBoost and LightGBM don't give you this built-in. You'd need separate tools. In ClearGBM, it's part of the design."

---

## WRAP UP (20 seconds)

"ClearGBM. Gradient boosting from scratch. Competitive accuracy, interpretable by design, pure Python, zero dependencies.

The code is on GitHub — 2,500 lines, fully typed, 100% test coverage. Every prediction can be explained."

---

## TOTAL: ~5 minutes
