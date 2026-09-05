# Regression Models, Built From Scratch

Nine regression models, one real dataset, every step explained for someone who has
never written Python.

This repository contains a Jupyter notebook that builds up from a single straight line
to a 300-tree random forest, predicting house prices from **2,930 real sales in Ames,
Iowa (2006–2010)**. Each model exists to fix a specific problem the previous one had,
and every model is trained, scored honestly on unseen data, and saved to disk.

---

## What's in here

| Path | What it is |
|---|---|
| `regression_models.ipynb` | **The main thing.** The full notebook, with all outputs and charts saved |
| `regression_models_handout.pdf` | The same notebook as a printable handout |
| `data/ames.csv` | The dataset — 2,930 houses × 82 columns |
| `models/*.joblib` | The nine trained models, ready to load and use |
| `scripts/get_data.py` | Re-downloads the dataset from its original source |
| `make_pdf.py` | Rebuilds the PDF handout from the notebook |
| `requirements.txt` | The Python packages you need |

---

## The models

| # | Model | The problem it solves |
|---|-------|----------------------|
| 1 | Simple Linear Regression | Draw one straight line through one feature |
| 2 | Multiple Linear Regression | One feature isn't enough — use twenty-seven |
| 3 | Polynomial Regression | Some relationships bend; straight lines can't |
| 4 | Ridge (L2) | Punish complexity so the model can't run away |
| 5 | Lasso (L1) | Punish complexity *and* delete useless features |
| 6 | Elastic Net | Ridge and Lasso, blended |
| 7 | Support Vector Regression | Fit a tolerance tube instead of chasing every point |
| 8 | Decision Tree | Forget equations — ask yes/no questions |
| 9 | Random Forest | One tree is a fragile opinion. Average 300 |

Along the way it covers **MAE, MSE, RMSE and R²**, the train/test split, overfitting,
cross-validation, one-hot encoding, imputation, scikit-learn pipelines, and data leakage.

---

## Running it

You need Python 3.9 or newer.

```bash
# 1. Get the code
git clone https://github.com/YOUR-USERNAME/regression-models-handout.git
cd regression-models-handout

# 2. Install the packages
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook regression_models.ipynb
```

Then run the cells from top to bottom (`Shift + Enter` runs one cell). The whole
notebook takes about six minutes to execute. Cells depend on the ones above them, so if
something breaks, use **Kernel → Restart & Run All** rather than re-running one cell.

### Try it live in the browser (Streamlit)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

This opens an interactive form in your browser — pick a model from the sidebar, fill in
a house's details, and get a price prediction instantly. To put it online for free
(so you can share a link instead of asking people to install anything), push this repo
to GitHub, then go to [share.streamlit.io](https://share.streamlit.io), sign in with
GitHub, click **New app**, pick this repo and set the main file to `streamlit_app.py`.

### Using a saved model without running anything

```python
import joblib, pandas as pd

model = joblib.load("models/09_random_forest.joblib")

house = pd.DataFrame([{
    "Gr Liv Area": 1800, "Total Bsmt SF": 900, "1st Flr SF": 1000, "Lot Area": 9000,
    "Overall Qual": 7, "Overall Cond": 5, "Year Built": 2001, "Year Remod/Add": 2001,
    "Garage Cars": 2, "Garage Area": 500, "Full Bath": 2, "Half Bath": 1,
    "TotRms AbvGrd": 7, "Fireplaces": 1, "Mas Vnr Area": 120,
    "Wood Deck SF": 150, "Open Porch SF": 40,
    "Neighborhood": "CollgCr", "House Style": "2Story", "Bldg Type": "1Fam",
    "Exter Qual": "Gd", "Kitchen Qual": "Gd", "Bsmt Qual": "Gd",
    "Heating QC": "Ex", "Central Air": "Y", "Foundation": "PConc",
    "Garage Type": "Attchd",
}])

print(f"${model.predict(house)[0]:,.0f}")
```

Each saved model is a complete pipeline — it does its own missing-value filling, scaling
and category encoding. You hand it raw columns and it handles the rest.

---

## Never used GitHub before? Start here.

GitHub is a website where you store code so other people can see it, and so you have a
history of every change you made. Putting this project there gives you a link you can
put on a CV or send to a student. This section assumes you have never used it.

### What the words mean

| Word | What it actually means |
|---|---|
| **Repository** (or "repo") | One project's folder, living on GitHub |
| **Git** | The program on your computer that tracks changes to files |
| **Commit** | A saved snapshot of your files, with a note saying what changed |
| **Push** | Send your commits from your computer up to GitHub |
| **Clone** | Copy a repository from GitHub down to your computer |
| **README.md** | The file you are reading. GitHub displays it on the repo's front page |

### Path A — the easy way, no command line (10 minutes)

Good enough for this project. Do this if you have never opened a terminal.

1. Go to **[github.com](https://github.com)** and create a free account.
2. Click the **+** in the top-right corner → **New repository**.
3. Fill in:
   - **Repository name:** `regression-models-handout`
   - **Description:** `Nine regression models built step by step on the Ames Housing data`
   - Choose **Public** (so you can share the link).
   - **Do not** tick "Add a README file" — this project already has one.
4. Click **Create repository**.
5. On the next page click **uploading an existing file**.
6. Drag this entire project folder into the browser window and wait for the upload.
7. In the **Commit changes** box at the bottom, type `Initial commit`, then click
   **Commit changes**.

Done. Your project is live at `github.com/YOUR-USERNAME/regression-models-handout`.

> **One catch:** GitHub's web uploader refuses files over 25 MB. The random forest model
> is the only file that gets close. If it is rejected, upload everything else and use
> Path B for that one file — or simply delete it and let people regenerate it by running
> the notebook.

### Path B — the command line (the way you'll eventually want to work)

**Step 1 — install Git.**

- **Windows:** download from [git-scm.com/download/win](https://git-scm.com/download/win),
  run the installer, accept every default. This gives you a program called **Git Bash** —
  use that as your terminal.
- **macOS:** open Terminal and type `git --version`. If Git isn't installed, macOS offers
  to install it. Say yes.
- **Linux:** `sudo apt install git`

**Step 2 — tell Git who you are.** Once per computer, ever:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

Use the same email as your GitHub account.

**Step 3 — create the empty repository on GitHub.** Follow steps 1–4 of Path A above,
then leave that browser tab open. It shows you a URL like
`https://github.com/YOUR-USERNAME/regression-models-handout.git`.

**Step 4 — push your files.** In a terminal, navigate into this project folder
(`cd` followed by the folder path), then:

```bash
git init                    # start tracking this folder
git add .                   # stage every file (the . means "everything here")
git commit -m "Initial commit"      # save a snapshot with a message
git branch -M main                  # name the main line of work "main"
git remote add origin https://github.com/YOUR-USERNAME/regression-models-handout.git
git push -u origin main             # send it to GitHub
```

Replace `YOUR-USERNAME` with your actual username.

**Step 5 — the password isn't your password.** When Git asks you to log in, GitHub will
reject your account password. You need a **personal access token**:

GitHub → your avatar → **Settings** → **Developer settings** (very bottom of the left
menu) → **Personal access tokens** → **Tokens (classic)** → **Generate new token
(classic)** → tick the **repo** checkbox → **Generate token**.

Copy the token immediately — GitHub shows it exactly once. Paste it when Git asks for a
password. Save it somewhere safe; you will need it again.

**Step 6 — after you change something later:**

```bash
git add .
git commit -m "Say what you changed"
git push
```

Those three lines are 95% of daily Git use.

### Things that go wrong, and what they mean

| What you see | What to do |
|---|---|
| `fatal: not a git repository` | You're in the wrong folder. `cd` into the project folder first |
| `Authentication failed` | Use a personal access token, not your password (Step 5) |
| `Updates were rejected` | Someone (or you, elsewhere) changed GitHub's copy. Run `git pull` first, then push |
| `File is too large` | GitHub refuses files over 100 MB. Delete it or add it to `.gitignore` |
| `nothing to commit, working tree clean` | Nothing changed since your last commit. Not an error |

### Two habits worth forming immediately

- **Never commit secrets.** No API keys, no passwords, no client data. Once something is
  pushed it is in the history forever, even if you delete the file afterwards.
- **Write commit messages a stranger could read.** `Fixed it` tells you nothing in three
  months. `Add cross-validation to the Ridge model` tells you everything.

### What `.gitignore` does

The `.gitignore` file in this project lists things Git should pretend don't exist —
Python's `__pycache__` cache folders, editor settings, `.DS_Store` files on macOS.
Without it your repository fills up with junk nobody needs. Note that this project
*deliberately* does not ignore `models/` and `data/`, because the trained models and the
dataset are the point.

---

## The data

Dean De Cock (2011), *"Ames, Iowa: Alternative to the Boston Housing Data as an End of
Semester Regression Project"*, **Journal of Statistics Education**, Vol. 19, No. 3.

2,930 residential property sales recorded by the Ames City Assessor's Office between
2006 and 2010, described by 82 columns. Obtained via the
[`AmesHousing`](https://github.com/topepo/AmesHousing) R package (Max Kuhn). This is the
dataset behind Kaggle's *House Prices: Advanced Regression Techniques* competition, and
it was published specifically to replace the older Boston Housing dataset, which is
retired because of an ethically indefensible feature.

Following De Cock's own recommendation, the notebook removes the five houses larger than
4,000 sq ft — three of them were incomplete sales and are not representative of the
market.

Run `python scripts/get_data.py` to regenerate `data/ames.csv` from source.

---

## Licence

The notebook, code and documentation here are free to use, modify and teach from.
The dataset is public and belongs to its original authors, cited above.
#   r e g r e s s i o n - m o d e l s - h a n d o u t 
#Amgad Farg 
 
