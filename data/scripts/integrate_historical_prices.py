from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import json 
import sys
#Configuration
BASE_DIR= Path(__file__).resolve().parent.parent
RAW_DIR= BASE_DIR/"raw"
Training_dir=BASE_DIR/"training_datasets"
Training_dir.mkdir(parents=True,exist_ok=True)
#Paramètres (similaires que prepare_dataset.py)

OBSERVATION_WINDOW_MINUTES=10
BUY_THRESHOLD=0.005
SELL_THRESHOLD=-0.005

class HistoricalPriceIntegrator:
    def __init__(self):
        self.news_df=None
        self.historical_prices_df=None
        self.current_prices_df=None
    
    def load_data(self):
        """Charge toutes les données possibles"""

        print("Loading all available data")

        #Charger les news avec sentiments
        news_file=RAW_DIR/"news_sentiment.jsonl"
        if news_file.exists():
            self.news_df=pd.read_json(news_file,lines=True)
            print(f"Loaded {len(self.news_df)} news with sentiment")
        else:
            print("No existing news sentiment file found ")
            return False
        
        historical_prices_file=RAW_DIR/"historical_stock_prices.jsonl"
        if historical_prices_file.exists():
            self.historical_prices_df=pd.read_json(historical_prices_file,lines=True)
            print(f"Loaded {len(self.historical_prices_df)} historical price records")
        else: 
            print("File not found run hisotrical stock price before")
        
        current_prices_file= RAW_DIR/"stock_prices.jsonl"
        if current_prices_file.exists():
            self.current_prices_df=pd.read_json(current_prices_file,lines=True)
            print(f"Loaded {len(self.current_prices_df)} current price records")
        else:
            print("No current prices file found ")
        
        return self._combine_price_data()
    
    def _combine_price_data(self):
        """ Combine les prix historiques + prix actuels"""

        prices_df=[]
        if self.historical_prices_df is not None:
            hist_prices=self.historical_prices_df[["symbol","timestamp","price"]].copy()
            hist_prices['source']="historical"
            prices_df.append(hist_prices)
        
        if self.current_prices_df is not None:
            current_prices=self.current_prices_df[['symbol','timestamp','price']].copy()
            current_prices['source']='current'
            prices_df.append(current_prices)
        
        if not prices_df:
            print("No price data available")
            return False

        self.all_prices_df=pd.concat(prices_df,ignore_index=True)
        self.all_prices_df['timestamp']=pd.to_datetime(self.all_prices_df['timestamp'], errors="coerce").dt.tz_localize(None)

        self.all_prices_df=self.all_prices_df.dropna(subset=['symbol','timestamp','price'])
        self.all_prices_df['symbol']=self.all_prices_df['symbol'].astype(str).str.upper().str.strip()
        self.all_prices_df=self.all_prices_df.sort_values("timestamp").drop_duplicates(subset=['symbol', 'timestamp'],keep='last')
        print(f"✅ Combined price dataset: {len(self.all_prices_df):,} records")
        print(f"📊 Symbols available: {self.all_prices_df['symbol'].nunique()}")
        print(f"📅 Date range: {self.all_prices_df['timestamp'].min()} to {self.all_prices_df['timestamp'].max()}")
        
        return True
    
    
    def generate_expanded_training_samples(self) :

        """Génère les échantillons avec beaucoup plus de donnée prix"""

        self.news_df['timestamp']=pd.to_datetime(self.news_df['timestamp'], errors="coerce").dt.tz_localize(None)
        self.news_df=self.news_df.dropna(subset=['symbol','timestamp'])
        self.news_df['symbol']=self.news_df['symbol'].astype(str).str.upper().str.strip()

        self.news_df=self.news_df.drop_duplicates(subset=['symbol','timestamp'],keep='first')
        print(f"Processing {len(self.news_df)} unique news articles")

        training_samples=[]
        successful_matches=[]

        symbols_with_news=self.news_df['symbol'].unique()
        print(f" Symbols with news: {len(symbols_with_news)}")

        for symbol in symbols_with_news:
            symbol_news=self.news_df[self.news_df['symbol']==symbol].sort_values('timestamp')
            symbol_prices=self.all_prices_df[self.all_prices_df['symbol']==symbol].sort_values('timestamp')

            if symbol_prices.empty:
                print(f"No prices for {symbol}")
                continue
            
            print(f"Processing {symbol}: {len(symbol_news)} news × {len(symbol_prices)} prices")

            for _,news_row in symbol_news.iterrows():
                try:

                    news_timestamp=news_row["timestamp"]
                    prices_before = symbol_prices[symbol_prices['timestamp'] <= news_timestamp]
                    if prices_before.empty:
                        continue
                    
                    # Prix le plus proche avant la news
                    closest_price_idx = prices_before['timestamp'].idxmax()
                    p0 = prices_before.loc[closest_price_idx, 'price']
                    price_timestamp = prices_before.loc[closest_price_idx, 'timestamp']
                    
                    # Vérifier que le prix n'est pas trop ancien (max 2h)
                    time_diff = (news_timestamp - price_timestamp).total_seconds() / 3600
                    if time_diff > 2:  # Plus de 2h d'écart
                        continue
                    
                    # Prix futur (après fenêtre d'observation)
                    future_timestamp = news_timestamp + timedelta(minutes=OBSERVATION_WINDOW_MINUTES)
                    prices_after = symbol_prices[symbol_prices['timestamp'] >= future_timestamp]
                    
                    if prices_after.empty:
                        continue
                    
                    # Premier prix après la fenêtre
                    p1 = prices_after.iloc[0]['price']
                    variation = (p1 - p0) / p0
                    
                    # Logique de classification (même que prepare_dataset.py)
                    if variation >= BUY_THRESHOLD:
                        action = "BUY"
                    elif variation <= SELL_THRESHOLD:
                        action = "SELL"
                    else:
                        # Utiliser le sentiment pour les variations neutres
                        sentiment = news_row.get('sentiment_score', 0)
                        if sentiment > 0.1:
                            action = "BUY"
                        elif sentiment < -0.1:
                            action = "SELL"
                        else:
                            action = "HOLD"
                    
                    training_samples.append({
                        "symbol": symbol,
                        "text": news_row.get('content', ''),
                        "sentiment_score": news_row.get('sentiment_score', 0),
                        "price_now": p0,
                        "price_future": p1,
                        "variation": variation,
                        "action": action,
                        "news_timestamp": news_timestamp.isoformat(),
                        "price_timestamp": price_timestamp.isoformat()
                    })
                    
                    successful_matches = len(training_samples)
                    
                except Exception as e:
                    print(f"⚠️ Error processing sample for {symbol}: {e}")
                    continue
        
        print(f"✅ Generated {len(training_samples)} training samples ({successful_matches} successful matches)")
        
        return pd.DataFrame(training_samples)
    def merge_with_existing_dataset(self, new_samples_df):
        """Fusionne avec le dataset existant"""
        existing_file = Training_dir / "train.csv"
        
        if existing_file.exists():
            existing_df = pd.read_csv(existing_file)
            print(f"📊 Existing dataset: {len(existing_df)} samples")
            
            # Éviter les doublons basés sur symbole + contenu
            if 'text' in new_samples_df.columns and 'text' in existing_df.columns:
                new_samples_df['content_key'] = (
                    new_samples_df['symbol'] + '|' + new_samples_df['text'].str[:100]
                )
                existing_df['content_key'] = (
                    existing_df['symbol'] + '|' + existing_df['text'].str[:100] 
                )
                
                mask = ~new_samples_df['content_key'].isin(existing_df['content_key'])
                truly_new = new_samples_df[mask].drop('content_key', axis=1)
                
                if len(truly_new) > 0:
                    combined_df = pd.concat([
                        existing_df.drop('content_key', axis=1), 
                        truly_new
                    ], ignore_index=True)
                    print(f"➕ Added {len(truly_new)} new samples")
                else:
                    combined_df = existing_df.drop('content_key', axis=1)
                    print("ℹ️ No new unique samples to add")
            else:
                # Si pas de colonne text, juste combiner
                combined_df = pd.concat([existing_df, new_samples_df], ignore_index=True)
                print(f"➕ Added {len(new_samples_df)} new samples (no dedup check)")
        else:
            combined_df = new_samples_df
            print("📝 Creating new dataset file")
        
        return combined_df
    
    def run_integration(self):
        """Lance l'intégration complète"""
        print("🚀 HISTORICAL PRICE INTEGRATION PIPELINE")
        print("="*60)
        
        # 1. Charger toutes les données
        if not self.load_data():
            print("❌ Failed to load data")
            return
        
        # 2. Générer les échantillons avec plus de prix
        expanded_samples_df = self.generate_expanded_training_samples()
        
        if expanded_samples_df.empty:
            print("❌ No training samples generated")
            return
        
        # 3. Afficher les statistiques
        print("\n📊 EXPANDED DATASET STATISTICS")
        print("-" * 40)
        action_counts = expanded_samples_df['action'].value_counts()
        print(f"Total samples: {len(expanded_samples_df)}")
        for action, count in action_counts.items():
            percentage = (count / len(expanded_samples_df)) * 100
            print(f"{action}: {count} ({percentage:.1f}%)")
        
        # 4. Fusionner avec dataset existant
        final_dataset = self.merge_with_existing_dataset(expanded_samples_df)
        
        # 5. Sauvegarder
        output_file = Training_dir / "train.csv"
        final_dataset.to_csv(output_file, index=False)
        
        # Sauvegarder aussi une version avec metadata
        enhanced_file = Training_dir / "train_enhanced_with_historical.csv"
        expanded_samples_df.to_csv(enhanced_file, index=False)
        
        # 6. Rapport final
        self._generate_integration_report(final_dataset, expanded_samples_df)
        
        print(f"\n✅ Integration complete!")
        print(f"📁 Final dataset: {output_file}")
        print(f"📁 Enhanced dataset: {enhanced_file}")
        
        return final_dataset
    
    def _generate_integration_report(self, final_dataset, new_samples):
        """Génère un rapport d'intégration"""
        
        report = {
            "integration_date": datetime.now().isoformat(),
            "data_sources": {
                "news_articles": len(self.news_df) if self.news_df is not None else 0,
                "historical_price_records": len(self.historical_prices_df) if self.historical_prices_df is not None else 0,
                "current_price_records": len(self.current_prices_df) if self.current_prices_df is not None else 0,
                "combined_price_records": len(self.all_prices_df)
            },
            "dataset_expansion": {
                "new_samples_generated": len(new_samples),
                "final_dataset_size": len(final_dataset),
                "symbols_in_final": final_dataset['symbol'].nunique(),
                "new_action_distribution": new_samples['action'].value_counts().to_dict(),
                "final_action_distribution": final_dataset['action'].value_counts().to_dict()
            },
            "data_quality": {
                "price_coverage": {
                    "symbols_with_news": self.news_df['symbol'].nunique() if self.news_df is not None else 0,
                    "symbols_with_prices": self.all_prices_df['symbol'].nunique(),
                    "symbols_matched": new_samples['symbol'].nunique() if not new_samples.empty else 0
                },
                "time_coverage": {
                    "earliest_news": self.news_df['timestamp'].min().isoformat() if self.news_df is not None and not self.news_df.empty else None,
                    "latest_news": self.news_df['timestamp'].max().isoformat() if self.news_df is not None and not self.news_df.empty else None,
                    "earliest_price": self.all_prices_df['timestamp'].min().isoformat() if not self.all_prices_df.empty else None,
                    "latest_price": self.all_prices_df['timestamp'].max().isoformat() if not self.all_prices_df.empty else None
                }
            }
        }
        
        # Sauvegarder le rapport
        report_file = Training_dir / f"integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Affichage console
        print("\n" + "="*70)
        print("📊 HISTORICAL PRICE INTEGRATION REPORT")
        print("="*70)
        print(f"📰 News Articles Processed: {report['data_sources']['news_articles']:,}")
        print(f"📈 Historical Price Records: {report['data_sources']['historical_price_records']:,}")
        print(f"📊 Combined Price Records: {report['data_sources']['combined_price_records']:,}")
        print(f"🎯 New Training Samples: {report['dataset_expansion']['new_samples_generated']:,}")
        print(f"📋 Final Dataset Size: {report['dataset_expansion']['final_dataset_size']:,}")
        print(f"🏢 Symbols Matched: {report['data_quality']['price_coverage']['symbols_matched']}")
        
        print(f"\n📊 NEW SAMPLES ACTION DISTRIBUTION:")
        for action, count in report['dataset_expansion']['new_action_distribution'].items():
            percentage = (count / report['dataset_expansion']['new_samples_generated']) * 100 if report['dataset_expansion']['new_samples_generated'] > 0 else 0
            print(f"   {action}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n📊 FINAL DATASET ACTION DISTRIBUTION:")
        for action, count in report['dataset_expansion']['final_action_distribution'].items():
            percentage = (count / report['dataset_expansion']['final_dataset_size']) * 100 if report['dataset_expansion']['final_dataset_size'] > 0 else 0
            print(f"   {action}: {count:,} ({percentage:.1f}%)")
        
        print("="*70)
        print(f"📁 Report saved to: {report_file}")

def main():
    """Fonction principale"""
    print("🔗 Historical Price Integration for FinSentBot")
    print("="*50)
    
    integrator = HistoricalPriceIntegrator()
    
    # Vérifier les fichiers disponibles
    print("🔍 Checking available data files...")
    
    raw_dir = Path(__file__).resolve().parent.parent / "raw"
    files_status = {
        "news_sentiment.jsonl": (raw_dir / "news_sentiment.jsonl").exists(),
        "historical_stock_prices.jsonl": (raw_dir / "historical_stock_prices.jsonl").exists(), 
        "stock_prices.jsonl": (raw_dir / "stock_prices.jsonl").exists()
    }
    
    for filename, exists in files_status.items():
        status = "✅" if exists else "❌"
        print(f"{status} {filename}")
    
    if not files_status["news_sentiment.jsonl"]:
        print("\n❌ Critical: news_sentiment.jsonl not found!")
        print("   Make sure your news pipeline has run and generated sentiment scores.")
        return
    
    if not files_status["historical_stock_prices.jsonl"]:
        print("\n⚠️ Warning: historical_stock_prices.jsonl not found!")
        print("   Run the historical stock collector first:")
        print("   python historical_stock_collector.py")
        
        choice = input("\nContinue with only current prices? (y/n): ").lower().strip()
        if choice != 'y':
            print("Exiting. Run historical collection first.")
            return
    
    # Lancer l'intégration
    print("\n🚀 Starting integration...")
    final_dataset = integrator.run_integration()
    
    if final_dataset is not None and not final_dataset.empty:
        print(f"\n🎉 SUCCESS! Your dataset has been expanded!")
        print(f"📊 From ~36 samples → {len(final_dataset):,} samples")
        
        # Suggestion pour la suite
        print(f"\n📝 Next steps:")
        print(f"1. Review the enhanced dataset: data/training_datasets/train_enhanced_with_historical.csv")
        print(f"2. Train your model with the expanded dataset:")
        print(f"   python TradingLogic/SignalGenerator/train.py")
        print(f"3. Compare performance before/after expansion")
    else:
        print("\n❌ Integration failed or no new samples generated.")


if __name__ == "__main__":
    main()
