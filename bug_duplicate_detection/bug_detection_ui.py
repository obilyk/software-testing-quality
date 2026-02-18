import pickle
import random
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

CACHE_DIR = Path(".cache/")
SOURCES_FOLDER = Path("results")


class BugDuplicateDetector:
    def __init__(self, debug_mode=True):
        self._debug_mode = debug_mode
        self._dataset = None
        self._embeddings = None
        self._classifier = None
        self._vector_store = None
        if not SOURCES_FOLDER.exists():
            SOURCES_FOLDER.mkdir()

    def _get_embedding(self, id):
        return self._vector_store.get(ids=[str(id)], include=["embeddings"])["embeddings"][0]

    def _get_classifier_dataset(self, n=2000):
        duplicates_sample = self._dataset[~self._dataset["Duplicate"].isnull()].sample(n=n, random_state=42)
        non_duplicates_sample = self._dataset[self._dataset["Duplicate"].isnull()].sample(n=n, random_state=42)
        all_ids = duplicates_sample["Issue_id"].to_list() + non_duplicates_sample["Issue_id"].to_list()

        duplicates_sample["Duplicate_id"] = duplicates_sample["Duplicate"].apply(lambda x: random.choice(x))
        duplicates_sample["is_duplicate"] = True
        non_duplicates_sample["Duplicate_id"] = non_duplicates_sample["Issue_id"].apply(
            lambda _: random.choice(all_ids)
        )
        non_duplicates_sample["is_duplicate"] = False

        pairs_df = pd.concat([duplicates_sample, non_duplicates_sample])
        pairs_df["pair"] = pairs_df.apply(lambda x: tuple(sorted([x["Issue_id"], x["Duplicate_id"]])), axis=1)
        pairs_df = pairs_df.drop_duplicates(subset="pair", keep="last")
        pairs_df = pairs_df[pairs_df["Issue_id"] != pairs_df["Duplicate_id"]]
        pairs_df = pairs_df[["Issue_id", "Duplicate_id", "is_duplicate"]]
        pairs_df = pairs_df.reset_index(drop=True)

        return pairs_df

    def _split_classifier_dataset(self, pairs_dataset):
        text_concat = pairs_dataset[["Issue_id", "Duplicate_id"]].apply(
            lambda row: np.concatenate(
                [self._get_embedding(row["Issue_id"]), self._get_embedding(row["Duplicate_id"])]
            ),
            axis=1,
        )
        X = np.stack(text_concat.to_numpy())
        y = pairs_dataset["is_duplicate"].astype(int).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)
        return X_train, X_test, y_train, y_test

    def prepare_classifier(self):
        assert self._dataset is not None, "Dataset must be provided"
        pairs_dataset = self._get_classifier_dataset()
        X_train, X_test, y_train, y_test = self._split_classifier_dataset(pairs_dataset)
        self._classifier.fit(X_train, y_train)
        with open(SOURCES_FOLDER / "classifier.pkl", "wb") as f:
            pickle.dump(self._classifier, f)
        if self._debug_mode:
            y_pred = self._classifier.predict(X_test)
            print(classification_report(y_test, y_pred))
        # with open(SOURCES_FOLDER / "classifier.pkl", "rb") as f:
        #     self._classifier = pickle.load(f)

    def prepare_retriever(self):
        assert self._dataset is not None, "Dataset must be provided"
        self._vector_store = Chroma(
            collection_name="bug_duplicates",
            embedding_function=self._embeddings,
            persist_directory=str(SOURCES_FOLDER / "chroma_db"),
        )
        _ = self._vector_store.add_texts(texts=self._dataset["Text"].to_list(), ids=map(str, self._dataset["Issue_id"].to_list()))

    def _preprocess_data(self, data):
        data["Text"] = data["Title"] + " " + data["Description"]
        data["Duplicate"] = data["Duplicate"].apply(lambda x: list(map(int, x.split(";"))) if not pd.isnull(x) else x)
        # data = data.set_index("Issue_id")
        return data

    def load_data(self, file_path):
        dataset = pd.read_csv(file_path, index_col=0)
        self._dataset = self._preprocess_data(dataset)

    def setup_pipeline(self, pipeline_type="fast"):
        if pipeline_type == "accurate":
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2", cache_folder=str(CACHE_DIR)
            )
            self._classifier = SVC(kernel="rbf", probability=True)
        else:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=str(CACHE_DIR)
            )
            self._classifier = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )

    def get_duplicates(self, title, description, top_k=100, add=False):
        text = title + " " + description
        results = self._vector_store.similarity_search(
            text,
            k=top_k,
        )
        text_embedding = self._embeddings.embed_query(text)
        clf_input = [np.concatenate([text_embedding, self._get_embedding(res.id)]) for res in results]
        clf_input = np.stack(clf_input)
        clf_output = self._classifier.predict(clf_input)
        selected_duplicates = [int(results[i].id) for i, label in enumerate(clf_output) if label == 1]
        duplicates_subset = self._dataset[self._dataset["Issue_id"].isin(selected_duplicates)]
        if add:
            index = max(self._dataset.index) + 1
            self._dataset.loc[index] = [text, selected_duplicates]
            self._vector_store.add_texts(texts=[text], ids=[index])
        return duplicates_subset[["Issue_id", "Title", "Description"]]


class BugReportApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bug Duplicate Finder")

        self.detector = BugDuplicateDetector()  # Initialize detector
        self.dataset = None

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both")

        self.create_setup_tab()
        self.create_find_duplicates_tab()

    def create_setup_tab(self):
        self.setup_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.setup_tab, text="System Setup")

        ttk.Label(
            self.setup_tab, text="Upload a dataset of bug reports to set up duplicate retrieval and classification."
        ).pack(pady=5)

        self.load_button = ttk.Button(self.setup_tab, text="Upload Dataset", command=self.load_dataset)
        self.load_button.pack(pady=10)

        ttk.Label(self.setup_tab, text="Dataset Preview:").pack(pady=5)

        # Dataset Viewer (Treeview)
        self.dataset_tree = ttk.Treeview(
            self.setup_tab, columns=("ID", "Title", "Description", "Duplicates"), show="headings", height=10
        )
        self.dataset_tree.heading("ID", text="ID")
        self.dataset_tree.heading("Title", text="Title")
        self.dataset_tree.heading("Description", text="Description")
        self.dataset_tree.heading("Duplicates", text="Duplicates")

        self.dataset_tree.pack(expand=True, fill="both", pady=5)

        # Scrollbar for dataset preview
        dataset_scrollbar = ttk.Scrollbar(self.setup_tab, orient="vertical", command=self.dataset_tree.yview)
        dataset_scrollbar.pack(side="right", fill="y")
        self.dataset_tree.configure(yscroll=dataset_scrollbar.set)

        ttk.Label(self.setup_tab, text="Select a model (affects retrieval and classification):").pack(pady=5)

        self.model_choice = tk.StringVar(value="mpnet_svm")
        ttk.Radiobutton(
            self.setup_tab, text="Accuracy (MPNet + SVM)", variable=self.model_choice, value="mpnet_svm"
        ).pack(anchor="w", padx=10, pady=2)
        ttk.Radiobutton(
            self.setup_tab, text="Performance (MiniLM + XGBoost)", variable=self.model_choice, value="minilm_xgb"
        ).pack(anchor="w", padx=10, pady=2)

        # Button to set up the system after loading the dataset
        self.setup_button = ttk.Button(self.setup_tab, text="Set Up System", command=self.setup_pipeline)
        self.setup_button.pack(pady=10)
        self.setup_button["state"] = tk.DISABLED  # Initially disabled until dataset is loaded

        self.progress = ttk.Progressbar(self.setup_tab, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)

        ttk.Label(self.setup_tab, text="Process Logs:").pack(pady=5)
        self.log_text = tk.Text(self.setup_tab, height=8, width=80, state=tk.DISABLED)
        self.log_text.pack(pady=5)

    def create_find_duplicates_tab(self):
        self.find_duplicates_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.find_duplicates_tab, text="Find Duplicates")

        ttk.Label(self.find_duplicates_tab, text="Enter bug information").pack(pady=5)
        ttk.Label(self.find_duplicates_tab, text="Bug Title:").pack()
        self.title_entry = ttk.Entry(self.find_duplicates_tab, width=50)
        self.title_entry.pack()

        ttk.Label(self.find_duplicates_tab, text="Bug Description:").pack()
        self.description_entry = tk.Text(self.find_duplicates_tab, width=60, height=5)
        self.description_entry.pack()

        self.analyze_button = ttk.Button(self.find_duplicates_tab, text="Analyze", command=self.find_duplicates)
        self.analyze_button.pack(pady=10)

        ttk.Label(self.find_duplicates_tab, text="Found Duplicates:").pack()
        self.result_tree = ttk.Treeview(
            self.find_duplicates_tab, columns=("ID", "Title", "Description"), show="headings"
        )
        self.result_tree.heading("ID", text="ID")
        self.result_tree.heading("Title", text="Title")
        self.result_tree.heading("Description", text="Description")
        self.result_tree.pack(expand=True, fill="both")

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.detector.load_data(file_path)
            self.display_dataset()
            self.setup_button["state"] = tk.NORMAL  # Enable setup button

            messagebox.showinfo("Dataset Loaded", "The dataset was successfully loaded.")

    def display_dataset(self):
        """Displays the first few rows of the loaded dataset in the UI."""
        for row in self.dataset_tree.get_children():
            self.dataset_tree.delete(row)

        if self.detector._dataset is not None:
            preview_data = self.detector._dataset.head(10)  # Show only first 10 rows

            for _, row in preview_data.iterrows():
                issue_id = row["Issue_id"]
                title = row["Title"]
                description = row["Description"]
                duplicates = row["Duplicate"]
                self.dataset_tree.insert("", "end", values=(issue_id, title, description, duplicates))

    def log_message(self, message):
        """Logs messages in the UI"""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.configure(state=tk.DISABLED)
        self.log_text.yview(tk.END)

    def setup_pipeline(self):
        self.progress["value"] = 0
        self.log_message("Starting system setup...")
        model_choice = self.model_choice.get()
        self.detector.setup_pipeline("accurate" if model_choice == "mpnet_svm" else "fast")

        self.progress["value"] = 10
        self.log_message("Loading vector representations...")
        self.detector.prepare_retriever()

        self.progress["value"] = 80
        self.log_message("Training classifier...")
        self.detector.prepare_classifier()

        self.progress["value"] = 100
        self.log_message("Setup complete!")

        messagebox.showinfo("Setup Complete", "The model is ready for use.")

    def find_duplicates(self):
        title = self.title_entry.get()
        description = self.description_entry.get("1.0", tk.END).strip()
        if not title or not description:
            messagebox.showerror("Error", "Please enter both a title and a description.")
            return

        duplicates = self.detector.get_duplicates(title, description)

        for row in self.result_tree.get_children():
            self.result_tree.delete(row)

        for _, row in duplicates.iterrows():
            self.result_tree.insert("", "end", values=(row["Issue_id"], row["Title"], row["Description"]))

        messagebox.showinfo("Analysis Complete", "Check the found duplicates.")


if __name__ == "__main__":
    root = tk.Tk()
    app = BugReportApp(root)
    root.mainloop()
