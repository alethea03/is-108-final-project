"""
IS 108 – Intelligence System Final Project
Business Intelligence Predictive Modeling Application
Problem: Customer Churn Prediction
Algorithms: KNN, SVM, ANN
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ─────────────────────────────────────────────
#  Color Palette
# ─────────────────────────────────────────────
BG        = "#1e1e2e"
SIDEBAR   = "#181825"
CARD      = "#313244"
ACCENT    = "#cba6f7"   # purple
GREEN     = "#a6e3a1"
RED       = "#f38ba8"
YELLOW    = "#f9e2af"
BLUE      = "#89b4fa"
TEXT      = "#cdd6f4"
SUBTEXT   = "#a6adc8"
WHITE     = "#ffffff"

FONTS = {
    "title":   ("Segoe UI", 18, "bold"),
    "heading": ("Segoe UI", 13, "bold"),
    "normal":  ("Segoe UI", 10),
    "small":   ("Segoe UI", 9),
    "mono":    ("Consolas", 9),
    "btn":     ("Segoe UI", 10, "bold"),
}


# ─────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────
class ChurnApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BI Predictive Modeling – Customer Churn")
        self.geometry("1280x780")
        self.minsize(1100, 700)
        self.configure(bg=BG)

        # shared state
        self.df_raw       = None
        self.df_processed = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.scaler   = None
        self.models   = {}      # {"KNN": fitted_model, ...}
        self.results  = {}      # {"KNN": {accuracy, precision, ...}, ...}

        self._build_ui()

    # ── Layout ──────────────────────────────
    def _build_ui(self):
        # Sidebar
        sidebar = tk.Frame(self, bg=SIDEBAR, width=200)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="🔮 ChurnUp", font=("Segoe UI", 15, "bold"),
                 bg=SIDEBAR, fg=ACCENT).pack(pady=(24, 4))
        tk.Label(sidebar, text="BI Predictive Modeling", font=FONTS["small"],
                 bg=SIDEBAR, fg=SUBTEXT).pack(pady=(0, 24))

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=16, pady=4)

        self.pages = {}
        nav_items = [
            ("📂  Dataset",      "dataset"),
            ("⚙️  Preprocessing", "preprocess"),
            ("🤖  Train Models",  "train"),
            ("📊  Evaluation",    "evaluate"),
            ("🔍  Predict",       "predict"),
        ]
        self.nav_btns = {}
        for label, key in nav_items:
            btn = tk.Button(sidebar, text=label, font=FONTS["normal"],
                            bg=SIDEBAR, fg=TEXT, bd=0, anchor="w",
                            padx=20, pady=10, cursor="hand2",
                            activebackground=CARD, activeforeground=ACCENT,
                            command=lambda k=key: self._show_page(k))
            btn.pack(fill="x")
            self.nav_btns[key] = btn

        # Main content area
        self.content = tk.Frame(self, bg=BG)
        self.content.pack(side="left", fill="both", expand=True)

        # Build each page
        self.pages["dataset"]    = DatasetPage(self.content, self)
        self.pages["preprocess"] = PreprocessPage(self.content, self)
        self.pages["train"]      = TrainPage(self.content, self)
        self.pages["evaluate"]   = EvaluatePage(self.content, self)
        self.pages["predict"]    = PredictPage(self.content, self)

        for page in self.pages.values():
            page.place(relwidth=1, relheight=1)

        self._show_page("dataset")

    def _show_page(self, key):
        for k, btn in self.nav_btns.items():
            btn.configure(bg=SIDEBAR if k != key else CARD,
                          fg=ACCENT if k == key else TEXT)
        self.pages[key].tkraise()
        self.pages[key].on_show()


# ─────────────────────────────────────────────
#  Helper Widgets
# ─────────────────────────────────────────────
def card(parent, **kw):
    f = tk.Frame(parent, bg=CARD, bd=0, **kw)
    return f

def label(parent, text, font=None, fg=TEXT, **kw):
    return tk.Label(parent, text=text, font=font or FONTS["normal"],
                    bg=parent.cget("bg"), fg=fg, **kw)

def accent_btn(parent, text, command, width=None):
    kw = dict(width=width) if width else {}
    return tk.Button(parent, text=text, font=FONTS["btn"],
                     bg=ACCENT, fg="#1e1e2e", bd=0, pady=8, padx=16,
                     cursor="hand2", relief="flat", command=command,
                     activebackground="#b4befe", **kw)

def ghost_btn(parent, text, command, width=None):
    kw = dict(width=width) if width else {}
    return tk.Button(parent, text=text, font=FONTS["btn"],
                     bg=CARD, fg=TEXT, bd=1, pady=7, padx=14,
                     cursor="hand2", relief="flat", command=command,
                     activebackground=BG, **kw)

def stat_card(parent, title, value, color=GREEN):
    f = tk.Frame(parent, bg=CARD, padx=16, pady=12)
    tk.Label(f, text=title, font=FONTS["small"], bg=CARD, fg=SUBTEXT).pack(anchor="w")
    tk.Label(f, text=value,  font=("Segoe UI", 20, "bold"), bg=CARD, fg=color).pack(anchor="w")
    return f


# ─────────────────────────────────────────────
#  Page 1 – Dataset
# ─────────────────────────────────────────────
class DatasetPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self._build()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=32, pady=(28, 0))
        label(hdr, "📂  Dataset", font=FONTS["title"], fg=WHITE).pack(side="left")
        accent_btn(hdr, "Load CSV / Excel", self._load).pack(side="right")

        # Stats row
        self.stats_frame = tk.Frame(self, bg=BG)
        self.stats_frame.pack(fill="x", padx=32, pady=16)

        # Table area
        tbl_card = card(self)
        tbl_card.pack(fill="both", expand=True, padx=32, pady=(0, 24))

        label(tbl_card, "Dataset Preview", font=FONTS["heading"], fg=ACCENT).pack(
            anchor="w", padx=16, pady=(12, 6))

        # Treeview + scrollbars
        tv_frame = tk.Frame(tbl_card, bg=CARD)
        tv_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background=CARD, foreground=TEXT,
                        fieldbackground=CARD, rowheight=24,
                        font=FONTS["small"])
        style.configure("Treeview.Heading", background=SIDEBAR,
                        foreground=ACCENT, font=FONTS["small"])
        style.map("Treeview", background=[("selected", ACCENT)],
                  foreground=[("selected", "#1e1e2e")])

        self.tree = ttk.Treeview(tv_frame, show="headings")
        vsb = ttk.Scrollbar(tv_frame, orient="vertical",   command=self.tree.yview)
        hsb = ttk.Scrollbar(tv_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tv_frame.grid_rowconfigure(0, weight=1)
        tv_frame.grid_columnconfigure(0, weight=1)

        # Info bar
        self.info_var = tk.StringVar(value="No dataset loaded.")
        label(tbl_card, "", textvariable=self.info_var,
              fg=SUBTEXT).pack(anchor="w", padx=16, pady=(0, 8))

    def _load(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files","*.csv"),("Excel files","*.xlsx *.xls"),("All","*")])
        if not path:
            return
        try:
            if path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            self.app.df_raw = df
            self._refresh()
            # reset downstream state
            self.app.df_processed = None
            self.app.models = {}
            self.app.results = {}
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _refresh(self):
        df = self.app.df_raw
        if df is None:
            return

        # Stats
        for w in self.stats_frame.winfo_children():
            w.destroy()
        churn_pct = (df["Churn"].str.strip().str.lower() == "yes").mean() * 100 \
                    if "Churn" in df.columns else 0
        items = [
            ("Rows",    f"{len(df):,}",      BLUE),
            ("Columns", f"{len(df.columns)}", YELLOW),
            ("Missing", f"{df.isnull().sum().sum()}", RED if df.isnull().sum().sum() else GREEN),
            ("Churn %", f"{churn_pct:.1f}%",  RED),
        ]
        for title, val, col in items:
            s = stat_card(self.stats_frame, title, val, col)
            s.pack(side="left", padx=(0, 12))

        # Table – show first 200 rows
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(df.columns)
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=110, minwidth=60, anchor="center")
        for _, row in df.head(200).iterrows():
            self.tree.insert("", "end", values=list(row))

        missing = df.isnull().sum().sum()
        self.info_var.set(
            f"Showing first 200 of {len(df):,} rows  |  "
            f"{len(df.columns)} columns  |  Missing values: {missing}")

    def on_show(self):
        if self.app.df_raw is not None:
            self._refresh()


# ─────────────────────────────────────────────
#  Page 2 – Preprocessing
# ─────────────────────────────────────────────
class PreprocessPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self._build()

    def _build(self):
        label(self, "⚙️  Data Preprocessing", font=FONTS["title"], fg=WHITE).pack(
            anchor="w", padx=32, pady=(28, 16))

        # Two columns
        row = tk.Frame(self, bg=BG)
        row.pack(fill="both", expand=True, padx=32, pady=(0, 24))

        # Left – options
        left = card(row)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        label(left, "Preprocessing Steps", font=FONTS["heading"], fg=ACCENT).pack(
            anchor="w", padx=16, pady=(12, 8))

        steps = [
            "✔  Drop customerID column (non-predictive)",
            "✔  Handle missing / blank values (drop rows)",
            "✔  Convert TotalCharges to numeric",
            "✔  Label-encode all categorical columns",
            "✔  Feature scaling (StandardScaler)",
            "✔  Train / Test split  (80 % / 20 %)",
        ]
        for s in steps:
            tk.Label(left, text=s, font=FONTS["normal"], bg=CARD,
                     fg=TEXT, anchor="w").pack(fill="x", padx=20, pady=3)

        # Split ratio slider
        tk.Frame(left, bg=CARD, height=1).pack(fill="x", padx=16, pady=10)
        label(left, "Test Split Ratio", font=FONTS["small"], fg=SUBTEXT).pack(
            anchor="w", padx=20)
        self.split_var = tk.DoubleVar(value=0.2)
        sl = ttk.Scale(left, from_=0.1, to=0.4, variable=self.split_var,
                       orient="horizontal", length=200)
        sl.pack(anchor="w", padx=20, pady=4)
        self.split_lbl = label(left, "20 %", fg=YELLOW)
        self.split_lbl.pack(anchor="w", padx=20)
        self.split_var.trace_add("write",
            lambda *_: self.split_lbl.configure(
                text=f"{self.split_var.get()*100:.0f} %"))

        accent_btn(left, "▶  Run Preprocessing", self._run).pack(
            anchor="w", padx=20, pady=16)

        # Right – log
        right = card(row)
        right.pack(side="left", fill="both", expand=True)
        label(right, "Processing Log", font=FONTS["heading"], fg=ACCENT).pack(
            anchor="w", padx=16, pady=(12, 6))

        self.log = tk.Text(right, font=FONTS["mono"], bg=BG, fg=GREEN,
                           bd=0, state="disabled", wrap="word")
        vsb = ttk.Scrollbar(right, command=self.log.yview)
        self.log.configure(yscrollcommand=vsb.set)
        self.log.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 12))
        vsb.pack(side="right", fill="y", pady=(0, 12), padx=(0, 8))

    def _log(self, msg, color=GREEN):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _run(self):
        if self.app.df_raw is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return

        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

        df = self.app.df_raw.copy()
        self._log("── Starting Preprocessing ──")
        self._log(f"Original shape: {df.shape}")

        # Drop ID
        if "customerID" in df.columns:
            df.drop(columns=["customerID"], inplace=True)
            self._log("✔  Dropped customerID")

        # Fix TotalCharges
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            self._log("✔  Converted TotalCharges to numeric")

        # Drop missing
        before = len(df)
        df.dropna(inplace=True)
        df = df[df.apply(lambda r: all(str(v).strip() != "" for v in r), axis=1)]
        self._log(f"✔  Dropped {before - len(df)} rows with missing/blank values")

        # Label encode categoricals
        le = LabelEncoder()
        cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))
        self._log(f"✔  Label-encoded {len(cat_cols)} categorical columns")

        # Features / target
        X = df.drop(columns=["Churn"])
        y = df["Churn"]
        self._log(f"✔  Features: {list(X.columns)}")
        self._log(f"✔  Target  : Churn  (classes: {sorted(y.unique())})")

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.app.scaler = scaler
        self._log("✔  Applied StandardScaler")

        # Split
        test_sz = self.split_var.get()
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_sz, random_state=42, stratify=y)
        self.app.X_train, self.app.X_test = X_train, X_test
        self.app.y_train, self.app.y_test = y_train, y_test
        self.app.df_processed = df
        self.app.feature_names = list(X.columns)

        self._log(f"✔  Train samples : {len(X_train)}")
        self._log(f"✔  Test  samples : {len(X_test)}")
        self._log("")
        self._log("✅  Preprocessing complete!", TEXT)

    def on_show(self): pass


# ─────────────────────────────────────────────
#  Page 3 – Train Models
# ─────────────────────────────────────────────
class TrainPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self._build()

    def _build(self):
        label(self, "🤖  Train Models", font=FONTS["title"], fg=WHITE).pack(
            anchor="w", padx=32, pady=(28, 16))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=32, pady=(0, 24))

        # ── KNN card ──
        self._model_card(body, "K-Nearest Neighbor (KNN)",
            [("n_neighbors", "k value", 5, 1, 20)],
            "knn", BLUE)

        # ── SVM card ──
        self._model_card(body, "Support Vector Machine (SVM)",
            [("C", "C (Regularization)", 1.0, 0.01, 10.0)],
            "svm", YELLOW)

        # ── ANN card ──
        self._model_card(body, "Artificial Neural Network (ANN)",
            [("max_iter", "Max Iterations", 300, 100, 1000)],
            "ann", GREEN)

        # Train All button
        accent_btn(self, "▶▶  Train All Models", self._train_all).pack(
            anchor="w", padx=32, pady=(8, 0))

        # Progress / status
        self.status_var = tk.StringVar(value="")
        label(self, "", textvariable=self.status_var, fg=ACCENT).pack(
            anchor="w", padx=32, pady=8)

        self.progress = ttk.Progressbar(self, mode="indeterminate", length=400)
        self.progress.pack(anchor="w", padx=32)

    def _model_card(self, parent, title, params, key, color):
        c = card(parent)
        c.pack(fill="x", pady=(0, 10))

        header = tk.Frame(c, bg=CARD)
        header.pack(fill="x", padx=16, pady=(10, 4))
        label(header, title, font=FONTS["heading"], fg=color).pack(side="left")

        self_ref = self
        for attr, lbl_txt, default, lo, hi in params:
            row = tk.Frame(c, bg=CARD)
            row.pack(fill="x", padx=20, pady=2)
            label(row, lbl_txt + ":", fg=SUBTEXT).pack(side="left")
            var = tk.DoubleVar(value=default)
            sl  = ttk.Scale(row, from_=lo, to=hi, variable=var,
                            orient="horizontal", length=180)
            sl.pack(side="left", padx=8)
            val_lbl = label(row, f"{default}", fg=color)
            val_lbl.pack(side="left")
            var.trace_add("write", lambda *_, v=var, l=val_lbl:
                          l.configure(text=f"{v.get():.2f}"))
            # store param
            if not hasattr(self, "_params"):
                self._params = {}
            self._params[f"{key}_{attr}"] = var

    def _train_all(self):
        if self.app.X_train is None:
            messagebox.showwarning("Not Ready", "Run preprocessing first.")
            return
        self.progress.start(10)
        self.status_var.set("Training models…")
        threading.Thread(target=self._do_train, daemon=True).start()

    def _do_train(self):
        results = {}

        # KNN
        self.status_var.set("⏳ Training KNN…")
        k = max(1, int(self._params.get("knn_n_neighbors", tk.DoubleVar(value=5)).get()))
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(self.app.X_train, self.app.y_train)
        results["KNN"] = self._eval(knn)
        self.app.models["KNN"] = knn

        # SVM
        self.status_var.set("⏳ Training SVM…")
        C = self._params.get("svm_C", tk.DoubleVar(value=1.0)).get()
        svm = SVC(C=C, kernel="rbf", probability=True, random_state=42)
        svm.fit(self.app.X_train, self.app.y_train)
        results["SVM"] = self._eval(svm)
        self.app.models["SVM"] = svm

        # ANN
        self.status_var.set("⏳ Training ANN…")
        iters = max(100, int(self._params.get("ann_max_iter", tk.DoubleVar(value=300)).get()))
        ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=iters,
                            random_state=42, early_stopping=True)
        ann.fit(self.app.X_train, self.app.y_train)
        results["ANN"] = self._eval(ann)
        self.app.models["ANN"] = ann

        self.app.results = results
        self.progress.stop()
        self.status_var.set("✅  All models trained! Go to Evaluation.")

    def _eval(self, model):
        y_pred = model.predict(self.app.X_test)
        return {
            "accuracy":  accuracy_score(self.app.y_test, y_pred),
            "precision": precision_score(self.app.y_test, y_pred, zero_division=0),
            "recall":    recall_score(self.app.y_test, y_pred, zero_division=0),
            "f1":        f1_score(self.app.y_test, y_pred, zero_division=0),
            "cm":        confusion_matrix(self.app.y_test, y_pred),
        }

    def on_show(self): pass


# ─────────────────────────────────────────────
#  Page 4 – Evaluation
# ─────────────────────────────────────────────
class EvaluatePage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=32, pady=(28, 0))
        label(hdr, "📊  Model Evaluation", font=FONTS["title"], fg=WHITE).pack(side="left")
        ghost_btn(hdr, "🔄 Refresh", self.on_show).pack(side="right")

        # Metrics table card
        mc = card(self)
        mc.pack(fill="x", padx=32, pady=16)
        label(mc, "Performance Metrics", font=FONTS["heading"], fg=ACCENT).pack(
            anchor="w", padx=16, pady=(12, 6))

        cols = ("Model", "Accuracy", "Precision", "Recall", "F1-Score")
        style = ttk.Style()
        style.configure("Eval.Treeview", background=CARD, foreground=TEXT,
                        fieldbackground=CARD, rowheight=28, font=FONTS["normal"])
        style.configure("Eval.Treeview.Heading", background=SIDEBAR,
                        foreground=ACCENT, font=FONTS["small"])
        self.metrics_tree = ttk.Treeview(mc, columns=cols, show="headings",
                                         height=4, style="Eval.Treeview")
        for col in cols:
            self.metrics_tree.heading(col, text=col)
            self.metrics_tree.column(col, anchor="center", width=140)
        self.metrics_tree.pack(fill="x", padx=12, pady=(0, 12))

        # Confusion matrices row
        cm_lbl = label(self, "Confusion Matrices", font=FONTS["heading"], fg=ACCENT)
        cm_lbl.pack(anchor="w", padx=32, pady=(4, 6))

        self.cm_frame = tk.Frame(self, bg=BG)
        self.cm_frame.pack(fill="x", padx=32)

        # Best model banner
        self.best_var = tk.StringVar(value="")
        label(self, "", textvariable=self.best_var, fg=YELLOW,
              font=FONTS["heading"]).pack(anchor="w", padx=32, pady=10)

    def _draw_cm(self, parent, name, cm, color):
        c = card(parent)
        c.pack(side="left", padx=(0, 16), pady=4)
        label(c, name, font=FONTS["heading"], fg=color).pack(padx=12, pady=(8, 4))

        labels = ["No Churn", "Churned"]
        grid = tk.Frame(c, bg=CARD)
        grid.pack(padx=12, pady=(0, 12))

        # header row
        tk.Label(grid, text="", width=10, bg=CARD).grid(row=0, column=0)
        for j, lbl in enumerate(labels):
            tk.Label(grid, text=f"Pred: {lbl}", font=FONTS["small"],
                     bg=CARD, fg=SUBTEXT, width=12).grid(row=0, column=j+1)
        for i, rl in enumerate(labels):
            tk.Label(grid, text=f"Act: {rl}", font=FONTS["small"],
                     bg=CARD, fg=SUBTEXT, width=10).grid(row=i+1, column=0)
            for j in range(2):
                val = cm[i][j]
                bg  = GREEN if i == j else RED
                tk.Label(grid, text=str(val), font=("Segoe UI", 13, "bold"),
                         bg=bg, fg="#1e1e2e", width=8, pady=8).grid(
                    row=i+1, column=j+1, padx=2, pady=2)

    def on_show(self):
        if not self.app.results:
            return

        # Metrics table
        self.metrics_tree.delete(*self.metrics_tree.get_children())
        for name, r in self.app.results.items():
            self.metrics_tree.insert("", "end", values=(
                name,
                f"{r['accuracy']*100:.2f}%",
                f"{r['precision']*100:.2f}%",
                f"{r['recall']*100:.2f}%",
                f"{r['f1']*100:.2f}%",
            ))

        # Confusion matrices
        for w in self.cm_frame.winfo_children():
            w.destroy()
        colors = {"KNN": BLUE, "SVM": YELLOW, "ANN": GREEN}
        for name, r in self.app.results.items():
            self._draw_cm(self.cm_frame, name, r["cm"], colors[name])

        # Best model
        best = max(self.app.results, key=lambda k: self.app.results[k]["f1"])
        f1   = self.app.results[best]["f1"]
        self.best_var.set(f"🏆  Best Model: {best}  (F1 = {f1*100:.2f}%)")


# ─────────────────────────────────────────────
#  Page 5 – Predict
# ─────────────────────────────────────────────
class PredictPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self._build()

    def _build(self):
        label(self, "🔍  Make a Prediction", font=FONTS["title"], fg=WHITE).pack(
            anchor="w", padx=32, pady=(28, 8))
        label(self, "Enter customer details below to predict churn.",
              fg=SUBTEXT).pack(anchor="w", padx=32, pady=(0, 16))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=32, pady=(0, 24))

        # Input form card
        form_card = card(body)
        form_card.pack(side="left", fill="both", expand=True, padx=(0, 16))
        label(form_card, "Customer Features", font=FONTS["heading"], fg=ACCENT).pack(
            anchor="w", padx=16, pady=(12, 8))

        # Scrollable form
        canvas  = tk.Canvas(form_card, bg=CARD, highlightthickness=0)
        vsb     = ttk.Scrollbar(form_card, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.form_inner = tk.Frame(canvas, bg=CARD)
        canvas.create_window((0, 0), window=self.form_inner, anchor="nw")
        self.form_inner.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Result card
        res_card = card(body)
        res_card.pack(side="left", fill="both", expand=True)
        label(res_card, "Prediction Results", font=FONTS["heading"], fg=ACCENT).pack(
            anchor="w", padx=16, pady=(12, 8))

        self.result_frame = tk.Frame(res_card, bg=CARD)
        self.result_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Field definitions (name, type, options/range)
        self.field_defs = [
            ("gender",           "combo", ["Male", "Female"]),
            ("SeniorCitizen",    "combo", ["0", "1"]),
            ("Partner",          "combo", ["Yes", "No"]),
            ("Dependents",       "combo", ["Yes", "No"]),
            ("tenure",           "entry", None),
            ("PhoneService",     "combo", ["Yes", "No"]),
            ("MultipleLines",    "combo", ["Yes", "No", "No phone service"]),
            ("InternetService",  "combo", ["DSL", "Fiber optic", "No"]),
            ("OnlineSecurity",   "combo", ["Yes", "No", "No internet service"]),
            ("OnlineBackup",     "combo", ["Yes", "No", "No internet service"]),
            ("DeviceProtection", "combo", ["Yes", "No", "No internet service"]),
            ("TechSupport",      "combo", ["Yes", "No", "No internet service"]),
            ("StreamingTV",      "combo", ["Yes", "No", "No internet service"]),
            ("StreamingMovies",  "combo", ["Yes", "No", "No internet service"]),
            ("Contract",         "combo", ["Month-to-month", "One year", "Two year"]),
            ("PaperlessBilling", "combo", ["Yes", "No"]),
            ("PaymentMethod",    "combo", ["Electronic check", "Mailed check",
                                           "Bank transfer (automatic)",
                                           "Credit card (automatic)"]),
            ("MonthlyCharges",   "entry", None),
            ("TotalCharges",     "entry", None),
        ]
        self.field_vars = {}
        self._fields_built = False

    def _build_fields(self):
        for w in self.form_inner.winfo_children():
            w.destroy()
        self.field_vars = {}
        for name, ftype, opts in self.field_defs:
            row = tk.Frame(self.form_inner, bg=CARD)
            row.pack(fill="x", padx=12, pady=3)
            tk.Label(row, text=name, font=FONTS["small"], bg=CARD,
                     fg=SUBTEXT, width=20, anchor="w").pack(side="left")
            if ftype == "combo":
                var = tk.StringVar(value=opts[0])
                cb  = ttk.Combobox(row, textvariable=var, values=opts,
                                   state="readonly", width=22, font=FONTS["small"])
                cb.pack(side="left")
            else:
                var = tk.StringVar(value="0")
                tk.Entry(row, textvariable=var, width=10, font=FONTS["small"],
                         bg=BG, fg=TEXT, insertbackground=TEXT, bd=0,
                         relief="flat").pack(side="left", ipady=4, padx=4)
            self.field_vars[name] = var

        # Predict button
        tk.Frame(self.form_inner, bg=CARD, height=8).pack()
        accent_btn(self.form_inner, "🔍  Predict Churn", self._predict).pack(
            anchor="w", padx=12, pady=8)
        self._fields_built = True

    def _predict(self):
        if not self.app.models:
            messagebox.showwarning("No Models", "Please train models first.")
            return

        # Build input row using the scaler's feature names
        features = getattr(self.app, "feature_names", [f[0] for f in self.field_defs])

        # Encode same way as preprocessing: LabelEncoder per column
        # We replicate by re-fitting on df_processed
        df_p = self.app.df_processed
        if df_p is None:
            messagebox.showerror("Error", "Preprocessing data not found.")
            return

        # Build a single-row df from inputs (raw values)
        raw = {}
        for name, ftype, _ in self.field_defs:
            val = self.field_vars[name].get()
            try:
                val = float(val) if ftype == "entry" else val
            except ValueError:
                val = 0.0
            raw[name] = [val]

        input_df = pd.DataFrame(raw)[features]

        # Re-encode categoricals to match training encoding
        df_orig = self.app.df_raw.copy()
        if "customerID" in df_orig.columns:
            df_orig.drop(columns=["customerID"], inplace=True)
        if "TotalCharges" in df_orig.columns:
            df_orig["TotalCharges"] = pd.to_numeric(df_orig["TotalCharges"], errors="coerce")
        df_orig.dropna(inplace=True)
        df_orig = df_orig[df_orig.apply(
            lambda r: all(str(v).strip() != "" for v in r), axis=1)]

        le = LabelEncoder()
        for col in input_df.select_dtypes(include=["object", "str"]).columns:
            if col in df_orig.columns:
                le.fit(df_orig[col].astype(str))
                try:
                    input_df[col] = le.transform(input_df[col].astype(str))
                except ValueError:
                    input_df[col] = 0
            else:
                input_df[col] = 0

        X_input = self.app.scaler.transform(input_df.values.reshape(1, -1))

        # Show results
        for w in self.result_frame.winfo_children():
            w.destroy()

        colors = {"KNN": BLUE, "SVM": YELLOW, "ANN": GREEN}
        for name, model in self.app.models.items():
            pred  = model.predict(X_input)[0]
            label_txt = "⚠️  WILL CHURN" if pred == 1 else "✅  WILL NOT CHURN"
            label_col = RED if pred == 1 else GREEN

            # Probability if available
            prob_txt = ""
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_input)[0]
                prob_txt = f"  (Churn prob: {prob[1]*100:.1f}%)"

            c = tk.Frame(self.result_frame, bg=BG, pady=6)
            c.pack(fill="x", padx=4, pady=4)
            tk.Label(c, text=name, font=FONTS["heading"],
                     bg=BG, fg=colors[name]).pack(anchor="w")
            tk.Label(c, text=label_txt + prob_txt,
                     font=("Segoe UI", 11, "bold"), bg=BG, fg=label_col).pack(anchor="w")
            ttk.Separator(self.result_frame).pack(fill="x", pady=2)

    def on_show(self):
        if not self._fields_built and self.app.df_processed is not None:
            self._build_fields()
        elif not self._fields_built:
            # show placeholder
            for w in self.form_inner.winfo_children():
                w.destroy()
            tk.Label(self.form_inner, text="\n  Run Preprocessing first to enable predictions.\n",
                     font=FONTS["normal"], bg=CARD, fg=SUBTEXT).pack(padx=12)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = ChurnApp()
    app.mainloop()
