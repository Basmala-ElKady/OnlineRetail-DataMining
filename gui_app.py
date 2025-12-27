import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
from io import StringIO

warnings.filterwarnings('ignore')

class DataMiningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Mining Projects - Data Cleaning & Analysis")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Data storage
        self.project1_df = None
        self.project1_cleaned_df = None
        self.project2_df = None
        self.project2_cleaned_df = None
        self.pca_model = None
        self.frequent_itemsets = None
        self.association_rules_df = None
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 10))
        style.configure('TLabel', font=('Arial', 10))
        
        self.setup_ui()
    
    def setup_ui(self):
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Project 1: PCA Tab
        self.project1_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.project1_frame, text="Project 1: PCA & Dimensionality Reduction")
        self.setup_project1_tab()
        
        # Project 2: Association Rules Tab
        self.project2_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.project2_frame, text="Project 2: Association Rule Mining")
        self.setup_project2_tab()
    
    def setup_project1_tab(self):
        """Setup Project 1 UI"""
        # Main frame with scrollbar
        main_frame = ttk.Frame(self.project1_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Project 1: Data Cleaning & PCA", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # File selection
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Dataset File:").pack(side=tk.LEFT, padx=5)
        self.project1_file_label = ttk.Label(file_frame, text="No file selected", foreground="red")
        self.project1_file_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.load_project1_file).pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="View Original Data", command=self.p1_view_original).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Clean Data", command=self.p1_clean_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="View Cleaned Data", command=self.p1_view_cleaned).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Normalization", command=self.p1_normalize).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Apply PCA", command=self.p1_apply_pca).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="PCA Visualization", command=self.p1_visualize_pca).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Export Results", command=self.p1_export_results).pack(side=tk.LEFT, padx=2)
        
        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Results & Information", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Text display with scrollbar
        scrollbar = ttk.Scrollbar(results_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.project1_text = tk.Text(results_frame, yscrollcommand=scrollbar.set, height=20, width=100)
        self.project1_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.project1_text.yview)
    
    def setup_project2_tab(self):
        """Setup Project 2 UI"""
        main_frame = ttk.Frame(self.project2_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Project 2: Transactional Data Cleaning & ARM", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # File selection
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Dataset File:").pack(side=tk.LEFT, padx=5)
        self.project2_file_label = ttk.Label(file_frame, text="No file selected", foreground="red")
        self.project2_file_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.load_project2_file).pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="View Original Data", command=self.p2_view_original).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Clean Data", command=self.p2_clean_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="View Cleaned Data", command=self.p2_view_cleaned).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Preprocess", command=self.p2_preprocess).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="ARM Analysis", command=self.p2_arm_analysis).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Visualizations", command=self.p2_visualize).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Export Results", command=self.p2_export_results).pack(side=tk.LEFT, padx=2)
        
        # Parameters frame
        param_frame = ttk.LabelFrame(control_frame, text="ARM Parameters", padding=5)
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="Min Support:").pack(side=tk.LEFT, padx=5)
        self.p2_min_support = ttk.Scale(param_frame, from_=0.01, to=0.5, orient=tk.HORIZONTAL)
        self.p2_min_support.set(0.01)
        self.p2_min_support.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.p2_support_label = ttk.Label(param_frame, text="0.01")
        self.p2_support_label.pack(side=tk.LEFT, padx=5)
        self.p2_min_support.config(command=lambda v: self.p2_support_label.config(text=f"{float(v):.2f}"))
        
        ttk.Label(param_frame, text="Min Confidence:").pack(side=tk.LEFT, padx=5)
        self.p2_min_conf = ttk.Scale(param_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL)
        self.p2_min_conf.set(0.5)
        self.p2_min_conf.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.p2_conf_label = ttk.Label(param_frame, text="0.50")
        self.p2_conf_label.pack(side=tk.LEFT, padx=5)
        self.p2_min_conf.config(command=lambda v: self.p2_conf_label.config(text=f"{float(v):.2f}"))
        
        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Results & Information", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Text display with scrollbar
        scrollbar = ttk.Scrollbar(results_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.project2_text = tk.Text(results_frame, yscrollcommand=scrollbar.set, height=20, width=100)
        self.project2_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.project2_text.yview)
    
    # ============= PROJECT 1: PCA =============
    
    def load_project1_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.project1_df = pd.read_csv(file_path)
                self.project1_file_label.config(text=f"Loaded: {file_path.split('/')[-1]}", foreground="green")
                self.log_p1(f"File loaded successfully: {file_path}\nShape: {self.project1_df.shape}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def p1_view_original(self):
        if self.project1_df is None:
            messagebox.showwarning("Warning", "Please load a file first")
            return
        
        self.log_p1("\n=== ORIGINAL DATA ===\n")
        self.log_p1(f"Shape: {self.project1_df.shape}\n")
        self.log_p1(f"Data Types:\n{self.project1_df.dtypes}\n")
        self.log_p1(f"\nFirst 5 Rows:\n{self.project1_df.head()}\n")
        self.log_p1(f"\nMissing Values:\n{self.project1_df.isnull().sum()}\n")
        self.log_p1(f"\nBasic Statistics:\n{self.project1_df.describe()}\n")
    
    def p1_clean_data(self):
        if self.project1_df is None:
            messagebox.showwarning("Warning", "Please load a file first")
            return
        
        self.log_p1("\n=== CLEANING DATA ===\n")
        self.project1_cleaned_df = self.project1_df.copy()
        
        # Handle missing values
        initial_shape = self.project1_cleaned_df.shape[0]
        self.log_p1(f"Initial rows: {initial_shape}")
        
        # Remove rows with missing values in numeric columns
        numeric_cols = self.project1_cleaned_df.select_dtypes(include=[np.number]).columns
        self.project1_cleaned_df = self.project1_cleaned_df.dropna(subset=numeric_cols)
        self.log_p1(f"After removing missing numeric values: {self.project1_cleaned_df.shape[0]} rows")
        
        # Remove negative and unrealistic values
        for col in numeric_cols:
            if col != 'Price':  # Price might have legitimate negative values
                self.project1_cleaned_df = self.project1_cleaned_df[self.project1_cleaned_df[col] >= 0]
        
        self.log_p1(f"After removing negative values: {self.project1_cleaned_df.shape[0]} rows")
        
        # Remove outliers using IQR
        for col in numeric_cols:
            Q1 = self.project1_cleaned_df[col].quantile(0.25)
            Q3 = self.project1_cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.project1_cleaned_df = self.project1_cleaned_df[
                (self.project1_cleaned_df[col] >= lower_bound) & 
                (self.project1_cleaned_df[col] <= upper_bound)
            ]
        
        self.log_p1(f"After removing outliers: {self.project1_cleaned_df.shape[0]} rows")
        
        # Handle categorical inconsistencies
        categorical_cols = self.project1_cleaned_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.project1_cleaned_df[col] = self.project1_cleaned_df[col].str.strip().str.lower()
        
        self.log_p1(f"Standardized categorical values")
        
        # Remove duplicates
        dup_count = self.project1_cleaned_df.duplicated().sum()
        self.project1_cleaned_df = self.project1_cleaned_df.drop_duplicates()
        self.log_p1(f"After removing duplicates: {dup_count} rows removed")
        
        self.log_p1(f"\nFinal cleaned shape: {self.project1_cleaned_df.shape}")
        self.log_p1(f"Rows removed: {initial_shape - self.project1_cleaned_df.shape[0]}")
        messagebox.showinfo("Success", "Data cleaning completed successfully")
    
    def p1_view_cleaned(self):
        if self.project1_cleaned_df is None:
            messagebox.showwarning("Warning", "Please clean data first")
            return
        
        self.log_p1("\n=== CLEANED DATA ===\n")
        self.log_p1(f"Shape: {self.project1_cleaned_df.shape}\n")
        self.log_p1(f"Data Types:\n{self.project1_cleaned_df.dtypes}\n")
        self.log_p1(f"\nFirst 5 Rows:\n{self.project1_cleaned_df.head()}\n")
        self.log_p1(f"\nMissing Values:\n{self.project1_cleaned_df.isnull().sum()}\n")
        self.log_p1(f"\nBasic Statistics:\n{self.project1_cleaned_df.describe()}\n")

    def p1_normalize(self):
        if self.project1_cleaned_df is None:
            messagebox.showwarning("Warning", "Please clean data first")
            return

        self.log_p1("\n=== NORMALIZATION ===\n")

        # Select only numeric columns
        numeric_data = self.project1_cleaned_df.select_dtypes(include=[np.number])

        if numeric_data.empty:
            messagebox.showerror("Error", "No numeric columns found")
            return

        self.log_p1(f"Columns to normalize: {list(numeric_data.columns)}")

        # Apply StandardScaler
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(numeric_data)

        self.log_p1(f"\nNormalization method: StandardScaler (Z-score normalization)")
        self.log_p1(f"Formula: (X - mean) / std_dev\n")
        self.log_p1(f"Normalized data shape: {normalized_data.shape}")
        self.log_p1(f"\nMean after normalization: {normalized_data.mean(axis=0)}")
        self.log_p1(f"Std deviation after normalization: {normalized_data.std(axis=0)}\n")

        # Store normalized data
        self.normalized_df = pd.DataFrame(normalized_data, columns=numeric_data.columns)

        self.log_p1("Data is ready for PCA")
        messagebox.showinfo("Success", "Normalization completed successfully")

    def p1_apply_pca(self):
        if not hasattr(self, 'normalized_df'):
            messagebox.showwarning("Warning", "Please normalize data first")
            return
        
        self.log_p1("\n=== APPLYING PCA ===\n")
        
        # Apply PCA
        self.pca_model = PCA()
        pca_data = self.pca_model.fit_transform(self.normalized_df)
        
        # Calculate explained variance
        explained_var = self.pca_model.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)
        
        self.log_p1(f"Number of features: {len(explained_var)}")
        self.log_p1(f"\nExplained Variance Ratio (first 10):")
        for i, var in enumerate(explained_var[:10]):
            self.log_p1(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")
        
        self.log_p1(f"\nCumulative Explained Variance (first 10):")
        for i, var in enumerate(cumsum_var[:10]):
            self.log_p1(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")
        
        # Find components needed for 95% variance
        n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
        self.log_p1(f"\nComponents needed for 95% variance: {n_components_95}")
        self.log_p1(f"Variance explained: {cumsum_var[n_components_95-1]*100:.2f}%")
        
        self.pca_data = pca_data
        self.log_p1("\nPCA applied successfully!")
        messagebox.showinfo("Success", "PCA applied successfully")

    def p1_visualize_pca(self):
        if self.pca_model is None:
            messagebox.showwarning("Warning", "Please apply PCA first")
            return

        # Create a figure with subplots
        fig = plt.figure(figsize=(14, 10))

        # Plot 1: Scree Plot (Variance Explained per Component)
        ax1 = plt.subplot(2, 2, 1)
        n_components_to_show = min(10, len(self.pca_model.explained_variance_ratio_))
        components = range(1, n_components_to_show + 1)
        explained_variance = self.pca_model.explained_variance_ratio_[:n_components_to_show]

        # Create bar chart for scree plot
        bars = ax1.bar(components, explained_variance, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Scree Plot')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(components)

        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot 2: Cumulative Explained Variance
        ax2 = plt.subplot(2, 2, 2)
        cumsum_var = np.cumsum(self.pca_model.explained_variance_ratio_[:n_components_to_show])

        ax2.plot(components, cumsum_var, 'bo-', linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(components)
        ax2.set_ylim([0, 1.1])

        # Add horizontal lines for thresholds
        thresholds = [0.80, 0.90, 0.95]
        colors = ['orange', 'green', 'red']
        for threshold, color in zip(thresholds, colors):
            ax2.axhline(y=threshold, color=color, linestyle='--', alpha=0.5, linewidth=1)
            ax2.text(n_components_to_show + 0.2, threshold, f'{threshold * 100:.0f}%',
                     va='center', ha='left', color=color, fontsize=9)

        # Plot 3: 2D PCA Projection
        ax3 = plt.subplot(2, 2, 3)

        pc1_var = self.pca_model.explained_variance_ratio_[0] * 100
        pc2_var = self.pca_model.explained_variance_ratio_[1] * 100

        # Create scatter plot
        scatter = ax3.scatter(self.pca_data[:, 0],
                              self.pca_data[:, 1],
                              c=range(self.pca_data.shape[0]),
                              cmap='viridis',
                              alpha=0.6,
                              s=20,
                              edgecolors='black',
                              linewidth=0.3)

        ax3.set_xlabel(f'Principal Component 1 ({pc1_var:.2f}%)')
        ax3.set_ylabel(f'Principal Component 2 ({pc2_var:.2f}%)')
        ax3.set_title('2D PCA Projection')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', alpha=0.3, linewidth=0.5)
        ax3.axvline(x=0, color='black', alpha=0.3, linewidth=0.5)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Data Point Index')

        # Plot 4: 3D PCA Projection
        ax4 = plt.subplot(2, 2, 4, projection='3d')

        pc3_var = self.pca_model.explained_variance_ratio_[2] * 100

        # Create 3D scatter plot
        scatter3d = ax4.scatter(self.pca_data[:, 0],
                                self.pca_data[:, 1],
                                self.pca_data[:, 2],
                                c=range(self.pca_data.shape[0]),
                                cmap='plasma',
                                alpha=0.6,
                                s=15,
                                depthshade=True)

        ax4.set_xlabel(f'PC1 ({pc1_var:.2f}%)')
        ax4.set_ylabel(f'PC2 ({pc2_var:.2f}%)')
        ax4.set_zlabel(f'PC3 ({pc3_var:.2f}%)')
        ax4.set_title('3D PCA Projection')

        # Set viewing angle
        ax4.view_init(elev=20, azim=30)

        # Add colorbar
        cbar3d = plt.colorbar(scatter3d, ax=ax4, pad=0.1)
        cbar3d.set_label('Data Point Index')

        # Adjust layout
        plt.tight_layout()

        # Create a new window for displaying the plot
        plot_window = tk.Toplevel(self.root)
        plot_window.title("PCA Visualizations")
        plot_window.geometry("1000x800")

        # Embed the matplotlib figure in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar for navigation (optional but useful)
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.log_p1("\nPCA visualization displayed successfully")

    def p1_export_results(self):
        if self.project1_cleaned_df is None:
            messagebox.showwarning("Warning", "Please process data first")
            return
        
        folder = filedialog.askdirectory(title="Select folder to save results")
        if not folder:
            return
        
        try:
            # Save cleaned data
            self.project1_cleaned_df.to_csv(f"{folder}/project1_cleaned_data.csv", index=False)
            
            # Save normalized data
            if hasattr(self, 'normalized_df'):
                self.normalized_df.to_csv(f"{folder}/project1_normalized_data.csv", index=False)
            
            # Save PCA results
            if self.pca_model is not None:
                pca_df = pd.DataFrame(
                    self.pca_data[:, :5],
                    columns=[f'PC{i+1}' for i in range(5)]
                )
                pca_df.to_csv(f"{folder}/project1_pca_results.csv", index=False)
                
                # Save variance ratios
                var_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(len(self.pca_model.explained_variance_ratio_))],
                    'Explained Variance': self.pca_model.explained_variance_ratio_,
                    'Cumulative Variance': np.cumsum(self.pca_model.explained_variance_ratio_)
                })
                var_df.to_csv(f"{folder}/project1_variance_ratios.csv", index=False)
            
            messagebox.showinfo("Success", f"Results saved to {folder}")
            self.log_p1(f"\nResults exported to {folder}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    # ============= PROJECT 2: ARM =============
    
    def load_project2_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.project2_df = pd.read_csv(file_path)
                self.project2_file_label.config(text=f"Loaded: {file_path.split('/')[-1]}", foreground="green")
                self.log_p2(f"File loaded successfully: {file_path}\nShape: {self.project2_df.shape}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def p2_view_original(self):
        if self.project2_df is None:
            messagebox.showwarning("Warning", "Please load a file first")
            return
        
        self.log_p2("\n=== ORIGINAL DATA (Project 2) ===\n")
        self.log_p2(f"Shape: {self.project2_df.shape}\n")
        self.log_p2(f"Data Types:\n{self.project2_df.dtypes}\n")
        self.log_p2(f"\nFirst 5 Rows:\n{self.project2_df.head()}\n")
        self.log_p2(f"\nMissing Values:\n{self.project2_df.isnull().sum()}\n")
        self.log_p2(f"\nBasic Statistics:\n{self.project2_df.describe()}\n")
    
    def p2_clean_data(self):
        if self.project2_df is None:
            messagebox.showwarning("Warning", "Please load a file first")
            return
        
        self.log_p2("\n=== CLEANING DATA (Project 2) ===\n")
        self.project2_cleaned_df = self.project2_df.copy()
        
        initial_shape = self.project2_cleaned_df.shape[0]
        self.log_p2(f"Initial rows: {initial_shape}")
        
        # Remove missing critical values
        critical_cols = ['Invoice', 'InvoiceDate']
        for col in critical_cols:
            if col in self.project2_cleaned_df.columns:
                self.project2_cleaned_df = self.project2_cleaned_df[self.project2_cleaned_df[col].notna()]
        
        self.log_p2(f"After removing missing critical values: {self.project2_cleaned_df.shape[0]} rows")
        
        # Remove duplicates
        dup_count = self.project2_cleaned_df.duplicated().sum()
        self.project2_cleaned_df = self.project2_cleaned_df.drop_duplicates()
        self.log_p2(f"After removing duplicates: {dup_count} rows removed")
        
        # Standardize dates if present
        if 'InvoiceDate' in self.project2_cleaned_df.columns:
            self.project2_cleaned_df['InvoiceDate'] = pd.to_datetime(
                self.project2_cleaned_df['InvoiceDate'], errors='coerce'
            )
            self.project2_cleaned_df = self.project2_cleaned_df[self.project2_cleaned_df['InvoiceDate'].notna()]
            self.log_p2(f"Standardized date format: {self.project2_cleaned_df.shape[0]} rows")
        
        # Standardize categorical values
        categorical_cols = self.project2_cleaned_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'InvoiceDate':
                self.project2_cleaned_df[col] = self.project2_cleaned_df[col].str.strip().str.lower()
        
        self.log_p2(f"Standardized categorical values")
        
        # Clean item descriptions (remove extra spaces, inconsistent separators)
        if 'Description' in self.project2_cleaned_df.columns:
            self.project2_cleaned_df['Description'] = self.project2_cleaned_df['Description'].str.strip().str.lower()
        
        self.log_p2(f"\nFinal cleaned shape: {self.project2_cleaned_df.shape}")
        self.log_p2(f"Rows removed: {initial_shape - self.project2_cleaned_df.shape[0]}")
        messagebox.showinfo("Success", "Data cleaning completed successfully")
    
    def p2_view_cleaned(self):
        if self.project2_cleaned_df is None:
            messagebox.showwarning("Warning", "Please clean data first")
            return
        
        self.log_p2("\n=== CLEANED DATA (Project 2) ===\n")
        self.log_p2(f"Shape: {self.project2_cleaned_df.shape}\n")
        self.log_p2(f"First 5 Rows:\n{self.project2_cleaned_df.head()}\n")
        self.log_p2(f"Missing Values:\n{self.project2_cleaned_df.isnull().sum()}\n")
    
    def p2_preprocess(self):
        if self.project2_cleaned_df is None:
            messagebox.showwarning("Warning", "Please clean data first")
            return
        
        self.log_p2("\n=== PREPROCESSING (Project 2) ===\n")
        
        # Create transaction basket
        if 'Invoice' in self.project2_cleaned_df.columns and 'Description' in self.project2_cleaned_df.columns:
            self.log_p2("Creating transaction basket matrix...")
            
            # Aggregate by invoice and description
            basket = self.project2_cleaned_df.groupby(['Invoice', 'Description']).size().unstack(fill_value=0)
            
            # Convert to binary matrix
            self.basket_sets = (basket > 0).astype(int)
            
            self.log_p2(f"Basket matrix shape: {self.basket_sets.shape}")
            self.log_p2(f"Number of transactions: {self.basket_sets.shape[0]}")
            self.log_p2(f"Number of unique items: {self.basket_sets.shape[1]}")
            
            # Calculate sparsity
            sparsity = 100 * (self.basket_sets.values != 0).sum() / self.basket_sets.size
            self.log_p2(f"Sparsity (% of non-zero cells): {sparsity:.2f}%")
            
            self.log_p2("\nPreprocessing complete. Ready for ARM analysis.")
            messagebox.showinfo("Success", "Preprocessing completed successfully")
        else:
            messagebox.showerror("Error", "Required columns (Invoice, Description) not found")
    
    def p2_arm_analysis(self):
        if not hasattr(self, 'basket_sets'):
            messagebox.showwarning("Warning", "Please preprocess data first")
            return
        
        try:
            min_support = float(self.p2_min_support.get())
            min_confidence = float(self.p2_min_conf.get())
            
            self.log_p2(f"\n=== ASSOCIATION RULE MINING ===\n")
            self.log_p2(f"Min Support: {min_support:.3f}")
            self.log_p2(f"Min Confidence: {min_confidence:.3f}\n")
            
            self.log_p2("Finding frequent itemsets...")
            self.frequent_itemsets = apriori(self.basket_sets, min_support=min_support, use_colnames=True, low_memory=True)
            
            self.log_p2(f"Frequent itemsets found: {len(self.frequent_itemsets)}")
            
            if len(self.frequent_itemsets) < 2:
                self.log_p2("Not enough frequent itemsets for rule generation")
                messagebox.showwarning("Warning", "Not enough frequent itemsets found. Try lower min_support")
                return
            
            self.log_p2("\nGenerating association rules...")
            self.association_rules_df = association_rules(self.frequent_itemsets, metric='confidence', min_threshold=min_confidence)
            
            self.log_p2(f"Association rules generated: {len(self.association_rules_df)}\n")
            
            if len(self.association_rules_df) > 0:
                # Sort by lift
                self.association_rules_df = self.association_rules_df.sort_values('lift', ascending=False)
                
                self.log_p2("=== TOP 10 RULES (by Lift) ===\n")
                for idx, (_, rule) in enumerate(self.association_rules_df.head(10).iterrows(), 1):
                    antecedents = ', '.join(list(rule['antecedents']))
                    consequents = ', '.join(list(rule['consequents']))
                    self.log_p2(f"\nRule {idx}:")
                    self.log_p2(f"  {antecedents} => {consequents}")
                    self.log_p2(f"  Support: {rule['support']:.4f}")
                    self.log_p2(f"  Confidence: {rule['confidence']:.4f}")
                    self.log_p2(f"  Lift: {rule['lift']:.4f}")
            
            messagebox.showinfo("Success", f"Found {len(self.association_rules_df)} association rules")
        except Exception as e:
            messagebox.showerror("Error", f"ARM analysis failed: {str(e)}")
    
    def p2_visualize(self):
        if self.association_rules_df is None or len(self.association_rules_df) == 0:
            messagebox.showwarning("Warning", "Please run ARM analysis first")
            return
        
        fig = Figure(figsize=(14, 10))
        
        # Plot 1: Confidence vs Support
        ax1 = fig.add_subplot(2, 2, 1)
        scatter = ax1.scatter(self.association_rules_df['support'], 
                             self.association_rules_df['confidence'],
                             c=self.association_rules_df['lift'], cmap='viridis', s=100, alpha=0.6)
        ax1.set_xlabel('Support')
        ax1.set_ylabel('Confidence')
        ax1.set_title('Support vs Confidence (color = Lift)')
        fig.colorbar(scatter, ax=ax1, label='Lift')
        
        # Plot 2: Confidence vs Lift
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(self.association_rules_df['confidence'], 
                   self.association_rules_df['lift'], alpha=0.6, s=100)
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Lift')
        ax2.set_title('Confidence vs Lift')
        ax2.axhline(y=1, color='r', linestyle='--', label='Lift=1')
        ax2.legend()
        
        # Plot 3: Support distribution
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.hist(self.association_rules_df['support'], bins=20, edgecolor='black')
        ax3.set_xlabel('Support')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Support Distribution')
        
        # Plot 4: Confidence distribution
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.hist(self.association_rules_df['confidence'], bins=20, edgecolor='black')
        ax4.set_xlabel('Confidence')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Confidence Distribution')
        
        fig.tight_layout()
        
        window = tk.Toplevel(self.root)
        window.title("Association Rules Visualizations")
        window.geometry("1000x800")
        
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.log_p2("\nVisualizations displayed")
    
    def p2_export_results(self):
        if self.project2_cleaned_df is None:
            messagebox.showwarning("Warning", "Please process data first")
            return
        
        folder = filedialog.askdirectory(title="Select folder to save results")
        if not folder:
            return
        
        try:
            # Save cleaned data
            self.project2_cleaned_df.to_csv(f"{folder}/project2_cleaned_data.csv", index=False)
            
            # Save frequent itemsets
            if self.frequent_itemsets is not None:
                self.frequent_itemsets.to_csv(f"{folder}/project2_frequent_itemsets.csv", index=False)
            
            # Save association rules
            if self.association_rules_df is not None and len(self.association_rules_df) > 0:
                rules_export = self.association_rules_df.copy()
                rules_export['antecedents'] = rules_export['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules_export['consequents'] = rules_export['consequents'].apply(lambda x: ', '.join(list(x)))
                rules_export.to_csv(f"{folder}/project2_association_rules.csv", index=False)
            
            messagebox.showinfo("Success", f"Results saved to {folder}")
            self.log_p2(f"\nResults exported to {folder}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    # ============= UTILITY METHODS =============
    
    def log_p1(self, message):
        self.project1_text.insert(tk.END, str(message) + '\n')
        self.project1_text.see(tk.END)
        self.root.update()
    
    def log_p2(self, message):
        self.project2_text.insert(tk.END, str(message) + '\n')
        self.project2_text.see(tk.END)
        self.root.update()


if __name__ == "__main__":
    root = tk.Tk()
    app = DataMiningGUI(root)
    root.mainloop()
