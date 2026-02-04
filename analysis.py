import os
import logging 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import lightkurve as lk 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix
logging.basicConfig(level=logging.INFO)
plt.style.use('ggplot')
sns.set_palette('viridis')

class StellarAnalyzer:
    def __init__(self,datapath):
        self.datapath=datapath
        self.df=None
        self.model=None
    def load_data(self):
        logger.info(f"loading from{self.datapath}")
        self.df=pd.read_csv(self.data_path, skiprows=1)
        return self.df
    def eda(self,output_dir='notebooks/plots'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        log.info("performing EDA")

        plt.figure(figsize=(8, 6))
        sns.countplot(x='class', data=self.df)
        plt.title('Distribution of Celestial Objects')
        plt.savefig(f"{output_dir}/class_distribution.png")

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='class', y='redshift', data=self.df)
        plt.title('Redshift Distribution by Class')
        plt.savefig(f"{output_dir}/redshift_boxplot.png")

        if all(col in self.df.columns for col in ['u', 'g', 'r']):
            self.df['u-g'] = self.df['u'] - self.df['g']
            self.df['g-r'] = self.df['g'] - self.df['r']
            
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x='u-g', y='g-r', hue='class', data=self.df, alpha=0.5, s=10)
            plt.title('Stellar Color-Color Diagram (u-g vs g-r)')
            plt.savefig(f"{output_dir}/color_color_diagram.png")

