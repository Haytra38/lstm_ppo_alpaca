#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Système de tracking détaillé de l'historique des trades
"""

import json
import pandas as pd
from datetime import datetime
import os

class TradeHistoryTracker:
    """
    Classe pour enregistrer et analyser l'historique détaillé des trades
    """
    
    def __init__(self):
        self.trades_history = []
        self.current_positions = {}
        self.trade_id_counter = 0
        
    def open_position(self, position_type, entry_price, entry_time, quantity, sl_price=None, tp_price=None, step=None):
        """
        Enregistre l'ouverture d'une position
        
        Args:
            position_type (str): 'Long' ou 'Short'
            entry_price (float): Prix d'entrée
            entry_time (datetime): Moment d'ouverture
            quantity (float): Quantité tradée
            sl_price (float): Prix de stop loss
            tp_price (float): Prix de take profit
            step (int): Étape du modèle
        
        Returns:
            int: ID du trade
        """
        self.trade_id_counter += 1
        trade_id = self.trade_id_counter
        
        trade_info = {
            'trade_id': trade_id,
            'position_type': position_type,
            'status': 'Open',
            'entry_price': entry_price,
            'entry_time': entry_time,
            'entry_step': step,
            'quantity': quantity,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'exit_price': None,
            'exit_time': None,
            'exit_step': None,
            'exit_reason': None,
            'profit_loss': None,
            'profit_loss_pct': None,
            'duration_steps': None,
            'max_favorable_price': entry_price,
            'max_adverse_price': entry_price,
            'max_favorable_pct': 0.0,
            'max_adverse_pct': 0.0
        }
        
        self.current_positions[trade_id] = trade_info
        return trade_id
    
    def update_position(self, trade_id, current_price, current_time=None, step=None):
        """
        Met à jour une position ouverte avec le prix actuel
        
        Args:
            trade_id (int): ID du trade
            current_price (float): Prix actuel
            current_time (datetime): Temps actuel
            step (int): Étape actuelle
        """
        if trade_id not in self.current_positions:
            return
        
        position = self.current_positions[trade_id]
        entry_price = position['entry_price']
        position_type = position['position_type']
        
        # Calculer le profit/perte non réalisé
        if position_type == 'Long':
            unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100
            # Mettre à jour les extremes
            if current_price > position['max_favorable_price']:
                position['max_favorable_price'] = current_price
                position['max_favorable_pct'] = unrealized_pnl_pct
            if current_price < position['max_adverse_price']:
                position['max_adverse_price'] = current_price
                position['max_adverse_pct'] = unrealized_pnl_pct
        else:  # Short
            unrealized_pnl_pct = (entry_price - current_price) / entry_price * 100
            # Mettre à jour les extremes
            if current_price < position['max_favorable_price']:
                position['max_favorable_price'] = current_price
                position['max_favorable_pct'] = unrealized_pnl_pct
            if current_price > position['max_adverse_price']:
                position['max_adverse_price'] = current_price
                position['max_adverse_pct'] = unrealized_pnl_pct
        
        position['current_price'] = current_price
        position['current_time'] = current_time
        position['current_step'] = step
        position['unrealized_pnl_pct'] = unrealized_pnl_pct
    
    def close_position(self, trade_id, exit_price, exit_time, exit_reason='Manual', step=None, transaction_fee=0.001):
        """
        Ferme une position et calcule les résultats
        
        Args:
            trade_id (int): ID du trade
            exit_price (float): Prix de sortie
            exit_time (datetime): Moment de fermeture
            exit_reason (str): Raison de la fermeture
            step (int): Étape de fermeture
            transaction_fee (float): Frais de transaction
        """
        if trade_id not in self.current_positions:
            return
        
        position = self.current_positions[trade_id]
        entry_price = position['entry_price']
        quantity = position['quantity']
        position_type = position['position_type']
        
        # Calculer le profit/perte
        if position_type == 'Long':
            profit_loss_pct = (exit_price - entry_price) / entry_price * 100
        else:  # Short
            profit_loss_pct = (entry_price - exit_price) / entry_price * 100
        
        # Soustraire les frais de transaction (entrée + sortie)
        profit_loss_pct -= (transaction_fee * 2 * 100)
        
        # Calculer le profit/perte en valeur absolue
        profit_loss = quantity * (exit_price - entry_price) if position_type == 'Long' else quantity * (entry_price - exit_price)
        profit_loss -= quantity * entry_price * transaction_fee * 2  # Frais
        
        # Calculer la durée
        duration_steps = step - position['entry_step'] if step and position['entry_step'] else None
        
        # Mettre à jour les informations de fermeture
        position.update({
            'status': 'Closed',
            'exit_price': exit_price,
            'exit_time': exit_time,
            'exit_step': step,
            'exit_reason': exit_reason,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'duration_steps': duration_steps
        })
        
        # Déplacer vers l'historique
        self.trades_history.append(position.copy())
        del self.current_positions[trade_id]
        
        return profit_loss_pct
    
    def get_trade_summary(self):
        """
        Génère un résumé des trades
        
        Returns:
            dict: Résumé des performances
        """
        if not self.trades_history:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'total_pnl': 0,
                'best_trade': 0,
                'worst_trade': 0
            }
        
        profitable_trades = [t for t in self.trades_history if t['profit_loss_pct'] > 0]
        losing_trades = [t for t in self.trades_history if t['profit_loss_pct'] <= 0]
        
        return {
            'total_trades': len(self.trades_history),
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(profitable_trades) / len(self.trades_history) * 100,
            'avg_profit': sum(t['profit_loss_pct'] for t in profitable_trades) / len(profitable_trades) if profitable_trades else 0,
            'avg_loss': sum(t['profit_loss_pct'] for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            'total_pnl': sum(t['profit_loss_pct'] for t in self.trades_history),
            'best_trade': max(t['profit_loss_pct'] for t in self.trades_history),
            'worst_trade': min(t['profit_loss_pct'] for t in self.trades_history),
            'avg_duration': sum(t['duration_steps'] for t in self.trades_history if t['duration_steps']) / len([t for t in self.trades_history if t['duration_steps']]) if any(t['duration_steps'] for t in self.trades_history) else 0
        }
    
    def get_trades_dataframe(self):
        """
        Retourne l'historique des trades sous forme de DataFrame
        
        Returns:
            pd.DataFrame: DataFrame avec tous les trades
        """
        if not self.trades_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades_history)
        
        # Formater les colonnes pour une meilleure lisibilité
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'])
        
        # Trier par ordre chronologique
        df = df.sort_values('trade_id')
        
        return df
    
    def save_trades_history(self, filename=None, results_dir="results"):
        """
        Sauvegarde l'historique des trades
        
        Args:
            filename (str): Nom du fichier (optionnel)
            results_dir (str): Dossier de sauvegarde
        
        Returns:
            str: Chemin du fichier sauvegardé
        """
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trades_history_{timestamp}.json"
        
        filepath = os.path.join(results_dir, filename)
        
        # Préparer les données pour la sauvegarde
        save_data = {
            'trades_history': self.trades_history,
            'current_positions': self.current_positions,
            'summary': self.get_trade_summary(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Convertir les datetime en string pour JSON
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj
        
        save_data = convert_datetime(save_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Historique des trades sauvegardé: {filepath}")
        return filepath
    
    def export_trades_csv(self, filename=None, results_dir="results"):
        """
        Exporte l'historique des trades en CSV
        
        Args:
            filename (str): Nom du fichier CSV
            results_dir (str): Dossier de sauvegarde
        
        Returns:
            str: Chemin du fichier CSV
        """
        df = self.get_trades_dataframe()
        
        if df.empty:
            print("❌ Aucun trade à exporter")
            return None
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trades_history_{timestamp}.csv"
        
        filepath = os.path.join(results_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"✅ Historique des trades exporté en CSV: {filepath}")
        return filepath
    
    def print_trades_summary(self):
        """
        Affiche un résumé détaillé des trades
        """
        summary = self.get_trade_summary()
        
        print("\n📊 RÉSUMÉ DES TRADES")
        print("=" * 25)
        print(f"Trades totaux: {summary['total_trades']}")
        print(f"Trades rentables: {summary['profitable_trades']}")
        print(f"Trades perdants: {summary['losing_trades']}")
        print(f"Taux de réussite: {summary['win_rate']:.1f}%")
        print(f"Profit moyen: {summary['avg_profit']:.2f}%")
        print(f"Perte moyenne: {summary['avg_loss']:.2f}%")
        print(f"P&L total: {summary['total_pnl']:.2f}%")
        print(f"Meilleur trade: {summary['best_trade']:.2f}%")
        print(f"Pire trade: {summary['worst_trade']:.2f}%")
        print(f"Durée moyenne: {summary['avg_duration']:.1f} steps")
    
    def print_recent_trades(self, n=10):
        """
        Affiche les derniers trades
        
        Args:
            n (int): Nombre de trades à afficher
        """
        if not self.trades_history:
            print("❌ Aucun trade dans l'historique")
            return
        
        recent_trades = self.trades_history[-n:]
        
        print(f"\n📋 DERNIERS {len(recent_trades)} TRADES")
        print("=" * 40)
        
        for trade in recent_trades:
            status_icon = "✅" if trade['profit_loss_pct'] > 0 else "❌"
            print(f"{status_icon} Trade #{trade['trade_id']} - {trade['position_type']}")
            print(f"   Entrée: {trade['entry_price']:.2f} (Step {trade['entry_step']})")
            print(f"   Sortie: {trade['exit_price']:.2f} (Step {trade['exit_step']}) - {trade['exit_reason']}")
            print(f"   P&L: {trade['profit_loss_pct']:.2f}% ({trade['duration_steps']} steps)")
            print(f"   Max favorable: {trade['max_favorable_pct']:.2f}%")
            print(f"   Max adverse: {trade['max_adverse_pct']:.2f}%")
            print()

# Fonction utilitaire pour intégrer facilement dans les environnements de trading
def create_trade_tracker():
    """
    Crée une nouvelle instance du tracker de trades
    
    Returns:
        TradeHistoryTracker: Instance du tracker
    """
    return TradeHistoryTracker()

if __name__ == "__main__":
    # Test du système
    print("🔍 Test du système de tracking des trades")
    
    tracker = TradeHistoryTracker()
    
    # Simuler quelques trades
    trade1 = tracker.open_position('Long', 100.0, datetime.now(), 1.0, sl_price=95.0, tp_price=110.0, step=1)
    tracker.update_position(trade1, 105.0, step=5)
    tracker.close_position(trade1, 108.0, datetime.now(), 'Take Profit', step=10)
    
    trade2 = tracker.open_position('Short', 200.0, datetime.now(), 0.5, sl_price=210.0, tp_price=180.0, step=15)
    tracker.update_position(trade2, 195.0, step=20)
    tracker.close_position(trade2, 185.0, datetime.now(), 'Take Profit', step=25)
    
    # Afficher les résultats
    tracker.print_trades_summary()
    tracker.print_recent_trades()
    
    # Sauvegarder
    tracker.save_trades_history()
    tracker.export_trades_csv()