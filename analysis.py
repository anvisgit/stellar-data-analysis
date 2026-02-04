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
    def train(self):
        logger.info("Training Classifier")
        drop_cols = ['objid', 'specobjid', 'class', 'run', 'rerun', 'camcol', 'field']
        x = self.df.drop(columns=[col for col in drop_cols if col in self.df.columns])
        if 'u-g' in self.df.columns:
            x['u-g'] = self.df['u-g']
            x['g-r'] = self.df['g-r']
        y = self.df['class']
        trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.2, random_state=42)
        self.model=RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(trainx,trainy)
        ypred=self.model.predict(testx)
        print("\nClassification Report:")
        print(classification_report(testy,ypred))

        imp=pd.series9self.model.feature_importances_, index=x.columns).sort_value(ascending=False)
        plt.figure(figsize=(12,6))
        imp.plot(kind='bar')
        plt.title('Feature Importance in Stellar Classification')
        plt.tight_layout()
        plt.savefig("notebooks/plots/feature_importance.png")
        return self.model

    def fetch(self, targetid,output_path='data/tess_light_curve.csv'):
        logger.info(f"Fetching TESS data for {target_id}")
        try:
            search_result = lk.search_lightcurve(target_id, mission='TESS')
            if len(search_result) == 0:
                print(f"No data found for {target_id}")
                search_result = lk.search_lightcurve('TIC 261136679', mission='TESS')
            
            lc = search_result[0].download()
            lc_df = pd.DataFrame({
                'time': lc.time.value,
                'flux': lc.flux.value,
                'flux_err': lc.flux_err.value
            })
            
            lc_df = lc_df.dropna()
            lc_df.to_csv(output_path, index=False)
            print(f"Real TESS data saved to {output_path}")
            return lc_df
            except Exception as e:
                print(f"Error fetching TESS data: {e}")
                return None
            lc_df = lc_df.dropna()
            lc_df.to_csv(output_path, index=False)
            print(f"Real TESS data saved to {output_path}")
            return lc_df
        except Exception as e:
            print(f"Error fetching TESS data: {e}")
            return None

    def timeseries_data(n_points=200, output_path='data/variable_star_lc.csv'):
        t = np.linspace(0, 50, n_points)
    period = 5.37
    amplitude = 1.2
    base_mag = 14.5
    flux = base_mag + amplitude * np.sin(2 * np.pi * t / period) + np.random.normal(0, 0.1, n_points)
    
    lc_df = pd.DataFrame({
        'mjd': t + 58000,
        'mag': flux,
        'error': [0.05] * n_points
    })
    
    lc_df.to_csv(output_path, index=False)
    
    plt.figure(figsize=(12, 4))
    plt.errorbar(lc_df['mjd'], lc_df['mag'], yerr=lc_df['error'], fmt='o', markersize=3, color='blue', alpha=0.6)
    plt.gca().invert_yaxis()
    plt.title('Simulated Pulsating Variable Star Light Curve')
    plt.savefig("notebooks/plots/light_curve.png")

if __name__ == "__main__":
    analyzer = StellarAnalyzer("data/sdss_stellar_data.csv")
    analyzer.load_data()
    analyzer.perform_eda()
    analyzer.train_classifier()
    analyzer.fetch_tess_data()
    generate_time_series_data()



        






