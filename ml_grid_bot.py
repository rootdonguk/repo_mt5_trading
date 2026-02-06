"""
🌟 AI 예측 그리드 트레이딩 봇 - 머신러닝 + 실시간 웹 대시보드
- 1분 후 가격 예측 (지도학습)
- 예측 기반 그리드 자동 배치
- 실시간 웹 대시보드 (총수익, 그리드 변동, 가격, 방향전환)
- H/L/Q/S 키 수동 청산 기능
"""
import MetaTrader5 as mt5
import time
from datetime import datetime, timedelta
import sys
import threading
import msvcrt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from collections import deque, defaultdict
import json

# ==================== 설정 ====================
GRID_CONFIG = {
    'symbol': 'BTCUSD',
    'magic_number': 999999,
    
    # 그리드 전략
    'grid_spacing': 0.01,
    'grid_levels': 100,
    'lot_per_order': 0.01,
    
    # 손실 관리
    'max_loss_per_position': 0.02,
    'flip_on_loss': True,
    
    # 수익 목표
    'take_profit_ticks': 0.01,
    
    # 기타
    'max_spread': 100,
    'check_interval': 0.3,
    'deviation': 20,
    
    # ML 설정
    'ml_lookback': 60,  # 60분 데이터로 학습
    'ml_retrain_interval': 300,  # 5분마다 재학습
    'prediction_confidence_threshold': 0.7,
}

class PricePredictor:
    """머신러닝 기반 가격 예측기"""
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.scaler = StandardScaler()
        self.price_history = deque(maxlen=lookback * 2)
        self.is_trained = False
        
    def add_price(self, price, timestamp):
        """가격 데이터 추가"""
        self.price_history.append({'price': price, 'time': timestamp})
    
    def create_features(self, prices):
        """특징 생성 (기술적 지표)"""
        # prices가 리스트인 경우 DataFrame으로 변환
        if isinstance(prices, list):
            df = pd.DataFrame({'price': prices})
        else:
            df = pd.DataFrame(prices)
            if 'price' not in df.columns:
                df.columns = ['price']
        
        # 이동평균
        df['ma_5'] = df['price'].rolling(5).mean()
        df['ma_10'] = df['price'].rolling(10).mean()
        df['ma_20'] = df['price'].rolling(20).mean()
        
        # 변동성
        df['volatility'] = df['price'].rolling(10).std()
        
        # 모멘텀
        df['momentum'] = df['price'].diff(5)
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 가격 변화율
        df['price_change'] = df['price'].pct_change()
        
        return df.dropna()
    
    def train(self):
        """모델 학습"""
        if len(self.price_history) < self.lookback + 10:
            return False
        
        prices = [p['price'] for p in self.price_history]
        df = self.create_features(prices)
        
        if len(df) < 30:
            return False
        
        # X: 현재 특징, y: 1분 후 가격
        X = df[['ma_5', 'ma_10', 'ma_20', 'volatility', 'momentum', 'rsi', 'price_change']].values[:-1]
        y = df['price'].values[1:]
        
        # 정규화
        X_scaled = self.scaler.fit_transform(X)
        
        # 학습
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        return True
    
    def predict_next_minute(self):
        """1분 후 가격 예측"""
        if not self.is_trained or len(self.price_history) < self.lookback:
            return None
        
        prices = [p['price'] for p in self.price_history]
        df = self.create_features(prices)
        
        if len(df) == 0:
            return None
        
        # 최신 특징
        latest_features = df[['ma_5', 'ma_10', 'ma_20', 'volatility', 'momentum', 'rsi', 'price_change']].iloc[-1:].values
        latest_scaled = self.scaler.transform(latest_features)
        
        # 예측
        predicted_price = self.model.predict(latest_scaled)[0]
        current_price = prices[-1]
        
        # 신뢰도 계산 (변동성 기반)
        volatility = df['volatility'].iloc[-1]
        confidence = 1.0 / (1.0 + volatility / current_price)
        
        return {
            'predicted_price': predicted_price,
            'current_price': current_price,
            'direction': 'up' if predicted_price > current_price else 'down',
            'change': predicted_price - current_price,
            'confidence': confidence
        }

class AIGridBot:
    def __init__(self, config):
        self.config = config
        self.grid_orders = {'buy': {}, 'sell': {}}
        self.active_positions = {}
        self.stats = {
            'total_profit': 0.0,
            'total_trades': 0,
            'grid_hits': 0,
            'flips': 0,
            'avoided_loss': 0.0,
            'start_time': datetime.now(),
            'predictions': [],
            'prediction_accuracy': 0.0,
            'start_balance': 0.0,  # 시작 잔고
            'start_equity': 0.0,   # 시작 증거금
        }
        self.center_price = None
        self.running = True
        self.manual_action = None
        
        # ML 예측기
        self.predictor = PricePredictor(config['ml_lookback'])
        self.last_retrain = time.time()
        self.last_prediction = None
        
        # 웹 대시보드 데이터
        self.dashboard_data = {
            'prices': deque(maxlen=100),
            'profits': deque(maxlen=100),
            'grid_changes': deque(maxlen=50),
            'flips': deque(maxlen=50),
            'predictions': deque(maxlen=50),
        }
        
    def connect_mt5(self):
        """MT5 연결"""
        print("\n" + "="*80)
        print("  🤖 AI 예측 그리드 봇 - 머신러닝 + 실시간 대시보드")
        print("="*80)
        
        if not mt5.initialize():
            print(f"❌ MT5 초기화 실패")
            return False
        
        account_info = mt5.account_info()
        if account_info is None:
            print("❌ 계좌 정보 없음")
            mt5.shutdown()
            return False
        
        # 시작 잔고 저장
        self.stats['start_balance'] = account_info.balance
        self.stats['start_equity'] = account_info.equity
        
        print("\n✓ MT5 연결 성공!")
        print(f"계좌: {account_info.login}")
        print(f"시작 잔고: ${account_info.balance:,.2f}")
        print(f"시작 증거금: ${account_info.equity:,.2f}")
        
        return True
    
    def clear_existing_positions_and_orders(self):
        """시작 전 모든 기존 포지션과 대기 주문 청산/취소"""
        # 기존 포지션 확인
        positions = mt5.positions_get(symbol=self.config['symbol'], magic=self.config['magic_number'])
        orders = mt5.orders_get(symbol=self.config['symbol'], magic=self.config['magic_number'])
        
        if not positions and not orders:
            print("\n✓ 기존 포지션 및 주문 없음")
            return True
        
        print(f"\n{'='*80}")
        print(f"  ⚠️  기존 포지션 및 주문 발견!")
        print(f"{'='*80}")
        
        if positions:
            print(f"포지션: {len(positions)}개")
            for pos in positions:
                pos_type = "매수" if pos.type == mt5.ORDER_TYPE_BUY else "매도"
                print(f"  - {pos_type} @ ${pos.price_open:,.2f} | Lot: {pos.volume}")
        
        if orders:
            print(f"대기 주문: {len(orders)}개")
        
        print(f"{'='*80}\n")
        
        answer = input("모든 기존 포지션과 주문을 삭제하시겠습니까? (y/n): ")
        
        if answer.lower() != 'y':
            print("❌ 사용자가 삭제를 취소했습니다.")
            return False
        
        print(f"\n{'='*80}")
        print(f"  🔄 기존 포지션 및 주문 정리 중...")
        print(f"{'='*80}\n")
        
        # 기존 포지션 청산
        if positions:
            current_price = self.get_current_price()
            if current_price:
                closed = 0
                for position in positions:
                    close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                    close_price = current_price['bid'] if close_type == mt5.ORDER_TYPE_SELL else current_price['ask']
                    
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": self.config['symbol'],
                        "volume": position.volume,
                        "type": close_type,
                        "position": position.ticket,
                        "price": close_price,
                        "deviation": self.config['deviation'],
                        "magic": self.config['magic_number'],
                        "comment": "CLEAR_EXISTING",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    
                    result = mt5.order_send(close_request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        closed += 1
                    time.sleep(0.05)
                
                print(f"✅ {closed}개 기존 포지션 청산 완료!")
        
        # 기존 대기 주문 취소
        if orders:
            canceled = 0
            for order in orders:
                remove_request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                }
                result = mt5.order_send(remove_request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    canceled += 1
                time.sleep(0.05)
            
            print(f"✅ {canceled}개 기존 대기 주문 취소 완료!")
        
        print(f"\n{'='*80}\n")
        return True
    
    def get_symbol_info(self):
        """심볼 정보"""
        symbol_info = mt5.symbol_info(self.config['symbol'])
        if symbol_info is None:
            print(f"❌ {self.config['symbol']} 심볼을 찾을 수 없습니다")
            return None
        
        if not symbol_info.visible:
            mt5.symbol_select(self.config['symbol'], True)
        
        return symbol_info
    
    def get_current_price(self):
        """현재가"""
        tick = mt5.symbol_info_tick(self.config['symbol'])
        if tick is None:
            return None
        return {'bid': tick.bid, 'ask': tick.ask, 'spread': tick.ask - tick.bid}
    
    def collect_historical_data(self):
        """과거 데이터 수집 (ML 학습용)"""
        print("\n📊 과거 데이터 수집 중...")
        
        # 1분봉 데이터 가져오기
        rates = mt5.copy_rates_from_pos(self.config['symbol'], mt5.TIMEFRAME_M1, 0, self.config['ml_lookback'] * 2)
        
        if rates is None or len(rates) == 0:
            print("❌ 과거 데이터 수집 실패")
            return False
        
        # 예측기에 데이터 추가
        for rate in rates:
            price = (rate['open'] + rate['close']) / 2
            timestamp = datetime.fromtimestamp(rate['time'])
            self.predictor.add_price(price, timestamp)
        
        print(f"✓ {len(rates)}개 데이터 수집 완료")
        
        # 초기 학습
        print("🧠 초기 모델 학습 중...")
        if self.predictor.train():
            print("✓ 모델 학습 완료!")
            return True
        else:
            print("❌ 모델 학습 실패")
            return False
    
    def predict_and_adjust_grid(self):
        """1분 후 가격 예측 및 그리드 조정"""
        prediction = self.predictor.predict_next_minute()
        
        if prediction is None:
            return None
        
        self.last_prediction = prediction
        self.stats['predictions'].append({
            'time': datetime.now(),
            'prediction': prediction
        })
        
        # 대시보드 데이터 업데이트
        self.dashboard_data['predictions'].append({
            'time': datetime.now().isoformat(),
            'predicted': prediction['predicted_price'],
            'current': prediction['current_price'],
            'direction': prediction['direction'],
            'confidence': prediction['confidence']
        })
        
        print(f"\n🔮 예측: {prediction['direction'].upper()} | "
              f"현재 ${prediction['current_price']:,.2f} → "
              f"예측 ${prediction['predicted_price']:,.2f} | "
              f"신뢰도 {prediction['confidence']:.2%}")
        
        # 신뢰도가 높으면 그리드 중심 조정
        if prediction['confidence'] >= self.config['prediction_confidence_threshold']:
            self.adjust_grid_center(prediction['predicted_price'])
        
        return prediction
    
    def adjust_grid_center(self, predicted_price):
        """예측 가격 기반 그리드 중심 조정"""
        if self.center_price is None:
            return
        
        # 예측 가격과 현재 중심의 차이
        price_diff = abs(predicted_price - self.center_price)
        
        # 차이가 그리드 간격의 5배 이상이면 재배치
        if price_diff > self.config['grid_spacing'] * 5:
            print(f"🔄 그리드 중심 재조정: ${self.center_price:,.2f} → ${predicted_price:,.2f}")
            
            # 기존 대기 주문 취소
            self.cancel_all_pending_orders()
            
            # 새 중심으로 그리드 재설정
            self.center_price = round(predicted_price, 2)
            self.setup_grid_orders()
            
            self.dashboard_data['grid_changes'].append({
                'time': datetime.now().isoformat(),
                'new_center': self.center_price,
                'reason': 'prediction'
            })
    
    def place_pending_order(self, order_type, price, lot_size):
        """지정가 주문"""
        if order_type == 'buy':
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": self.config['symbol'],
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY_LIMIT,
                "price": price,
                "deviation": self.config['deviation'],
                "magic": self.config['magic_number'],
                "comment": f"AI_GRID_BUY_{price:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
        else:
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": self.config['symbol'],
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_SELL_LIMIT,
                "price": price,
                "deviation": self.config['deviation'],
                "magic": self.config['magic_number'],
                "comment": f"AI_GRID_SELL_{price:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
        
        result = mt5.order_send(request)
        return result.order if result and result.retcode == mt5.TRADE_RETCODE_DONE else None
    
    def cancel_all_pending_orders(self):
        """모든 대기 주문 취소"""
        orders = mt5.orders_get(symbol=self.config['symbol'], magic=self.config['magic_number'])
        if orders:
            for order in orders:
                mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket})
            print(f"✓ {len(orders)}개 대기 주문 취소")
    
    def setup_grid(self):
        """초기 그리드 설정"""
        current_price = self.get_current_price()
        if not current_price:
            return False
        
        self.center_price = round((current_price['bid'] + current_price['ask']) / 2, 2)
        
        print(f"\n{'='*80}")
        print(f"  🎯 AI 그리드 설정")
        print(f"{'='*80}")
        print(f"중심 가격: ${self.center_price:,.2f}")
        print(f"간격: ${self.config['grid_spacing']}")
        print(f"레벨: {self.config['grid_levels']} × 2 = {self.config['grid_levels'] * 2}개")
        print(f"{'='*80}\n")
        
        return self.setup_grid_orders()
    
    def setup_grid_orders(self):
        """그리드 주문 배치"""
        print("📊 그리드 배치 중...")
        
        self.grid_orders = {'buy': {}, 'sell': {}}
        
        # 매수 주문
        for i in range(1, self.config['grid_levels'] + 1):
            buy_price = round(self.center_price - (i * self.config['grid_spacing']), 2)
            order_id = self.place_pending_order('buy', buy_price, self.config['lot_per_order'])
            if order_id:
                self.grid_orders['buy'][buy_price] = order_id
            if i % 20 == 0:
                print(f"  매수 {i}/{self.config['grid_levels']}")
            time.sleep(0.03)
        
        # 매도 주문
        for i in range(1, self.config['grid_levels'] + 1):
            sell_price = round(self.center_price + (i * self.config['grid_spacing']), 2)
            order_id = self.place_pending_order('sell', sell_price, self.config['lot_per_order'])
            if order_id:
                self.grid_orders['sell'][sell_price] = order_id
            if i % 20 == 0:
                print(f"  매도 {i}/{self.config['grid_levels']}")
            time.sleep(0.03)
        
        total = len(self.grid_orders['buy']) + len(self.grid_orders['sell'])
        print(f"\n✅ 그리드 완료: {total}개\n")
        
        return True
    
    def flip_position(self, position):
        """손실 포지션 방향 전환"""
        current_price = self.get_current_price()
        if not current_price:
            return False
        
        # 손실 계산
        if position.type == mt5.ORDER_TYPE_BUY:
            current_loss = (current_price['bid'] - position.price_open) * position.volume
            original_direction = "매수"
            new_direction = "매도"
            new_type = mt5.ORDER_TYPE_SELL
            new_price = current_price['bid']
            close_type = mt5.ORDER_TYPE_SELL
            close_price = current_price['bid']
        else:
            current_loss = (position.price_open - current_price['ask']) * position.volume
            original_direction = "매도"
            new_direction = "매수"
            new_type = mt5.ORDER_TYPE_BUY
            new_price = current_price['ask']
            close_type = mt5.ORDER_TYPE_BUY
            close_price = current_price['ask']
        
        # 청산
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config['symbol'],
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": close_price,
            "deviation": self.config['deviation'],
            "magic": self.config['magic_number'],
            "comment": "AI_FLIP_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        close_result = mt5.order_send(close_request)
        
        if not close_result or close_result.retcode != mt5.TRADE_RETCODE_DONE:
            return False
        
        # 반대 방향 진입
        flip_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config['symbol'],
            "volume": position.volume,
            "type": new_type,
            "price": new_price,
            "deviation": self.config['deviation'],
            "magic": self.config['magic_number'],
            "comment": "AI_FLIP_OPEN",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        flip_result = mt5.order_send(flip_request)
        
        if flip_result and flip_result.retcode == mt5.TRADE_RETCODE_DONE:
            self.stats['flips'] += 1
            self.stats['avoided_loss'] += abs(current_loss)
            
            print(f"\n🔄 방향 전환! {original_direction} → {new_direction} | 회피: ${abs(current_loss):.4f}")
            
            self.active_positions[flip_result.order] = {
                'type': new_type,
                'entry_price': new_price,
                'volume': position.volume,
                'flipped': True
            }
            
            if position.ticket in self.active_positions:
                del self.active_positions[position.ticket]
            
            # 대시보드 데이터
            self.dashboard_data['flips'].append({
                'time': datetime.now().isoformat(),
                'from': original_direction,
                'to': new_direction,
                'avoided_loss': abs(current_loss)
            })
            
            return True
        
        return False
    
    def check_and_manage_positions(self):
        """포지션 관리"""
        positions = mt5.positions_get(symbol=self.config['symbol'], magic=self.config['magic_number'])
        
        if not positions:
            return
        
        current_price = self.get_current_price()
        if not current_price:
            return
        
        for position in positions:
            # 새 포지션
            if position.ticket not in self.active_positions:
                self.active_positions[position.ticket] = {
                    'type': position.type,
                    'entry_price': position.price_open,
                    'volume': position.volume,
                    'flipped': False
                }
                self.stats['grid_hits'] += 1
                self.refill_grid(position.price_open, position.type)
            
            # 손익 계산
            if position.type == mt5.ORDER_TYPE_BUY:
                profit_loss = (current_price['bid'] - position.price_open) * position.volume
                close_price = current_price['bid']
            else:
                profit_loss = (position.price_open - current_price['ask']) * position.volume
                close_price = current_price['ask']
            
            # 손실 체크 및 방향 전환
            if self.config['flip_on_loss'] and profit_loss < -self.config['max_loss_per_position']:
                self.flip_position(position)
                continue
            
            # 수익 실현
            if profit_loss >= self.config['take_profit_ticks']:
                self.close_position_with_profit(position, close_price, profit_loss)
    
    def close_position_with_profit(self, position, close_price, profit):
        """수익 실현"""
        close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config['symbol'],
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": close_price,
            "deviation": self.config['deviation'],
            "magic": self.config['magic_number'],
            "comment": "AI_PROFIT",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(close_request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.stats['total_profit'] += profit
            self.stats['total_trades'] += 1
            
            if position.ticket in self.active_positions:
                del self.active_positions[position.ticket]
    
    def refill_grid(self, filled_price, filled_type):
        """그리드 재생성"""
        if filled_type == mt5.ORDER_TYPE_BUY:
            order_id = self.place_pending_order('buy', filled_price, self.config['lot_per_order'])
            if order_id:
                self.grid_orders['buy'][filled_price] = order_id
        else:
            order_id = self.place_pending_order('sell', filled_price, self.config['lot_per_order'])
            if order_id:
                self.grid_orders['sell'][filled_price] = order_id
    
    def analyze_positions(self):
        """포지션 분석"""
        positions = mt5.positions_get(symbol=self.config['symbol'], magic=self.config['magic_number'])
        
        if not positions:
            return {'profit_positions': [], 'loss_positions': [], 'total_profit': 0, 'total_loss': 0}
        
        current_price = self.get_current_price()
        if not current_price:
            return {'profit_positions': [], 'loss_positions': [], 'total_profit': 0, 'total_loss': 0}
        
        profit_positions = []
        loss_positions = []
        total_profit = 0
        total_loss = 0
        
        for position in positions:
            if position.type == mt5.ORDER_TYPE_BUY:
                pnl = (current_price['bid'] - position.price_open) * position.volume
            else:
                pnl = (position.price_open - current_price['ask']) * position.volume
            
            if pnl > 0:
                profit_positions.append({'position': position, 'profit': pnl})
                total_profit += pnl
            else:
                loss_positions.append({'position': position, 'loss': pnl})
                total_loss += pnl
        
        return {
            'profit_positions': profit_positions,
            'loss_positions': loss_positions,
            'total_profit': total_profit,
            'total_loss': total_loss
        }
    
    def close_profit_positions(self):
        """수익 포지션만 청산"""
        analysis = self.analyze_positions()
        
        if not analysis['profit_positions']:
            print("\n💡 수익 포지션이 없습니다.")
            return
        
        print(f"\n{'='*80}")
        print(f"  💙 수익 포지션 청산")
        print(f"{'='*80}")
        
        current_price = self.get_current_price()
        closed = 0
        
        for item in analysis['profit_positions']:
            position = item['position']
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            close_price = current_price['bid'] if close_type == mt5.ORDER_TYPE_SELL else current_price['ask']
            
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.config['symbol'],
                "volume": position.volume,
                "type": close_type,
                "position": position.ticket,
                "price": close_price,
                "deviation": self.config['deviation'],
                "magic": self.config['magic_number'],
                "comment": "MANUAL_PROFIT",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(close_request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1
            time.sleep(0.05)
        
        print(f"\n✅ {closed}개 수익 포지션 청산 완료!")
    
    def close_loss_positions(self):
        """손실 포지션만 청산"""
        analysis = self.analyze_positions()
        
        if not analysis['loss_positions']:
            print("\n💡 손실 포지션이 없습니다.")
            return
        
        print(f"\n{'='*80}")
        print(f"  ❤️ 손실 포지션 청산")
        print(f"{'='*80}")
        
        current_price = self.get_current_price()
        closed = 0
        
        for item in analysis['loss_positions']:
            position = item['position']
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            close_price = current_price['bid'] if close_type == mt5.ORDER_TYPE_SELL else current_price['ask']
            
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.config['symbol'],
                "volume": position.volume,
                "type": close_type,
                "position": position.ticket,
                "price": close_price,
                "deviation": self.config['deviation'],
                "magic": self.config['magic_number'],
                "comment": "MANUAL_LOSS",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(close_request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1
            time.sleep(0.05)
        
        print(f"\n✅ {closed}개 손실 포지션 청산 완료!")
    
    def close_all_positions(self):
        """모든 포지션 청산"""
        positions = mt5.positions_get(symbol=self.config['symbol'], magic=self.config['magic_number'])
        
        if not positions:
            print("\n💡 포지션이 없습니다.")
            return
        
        print(f"\n{'='*80}")
        print(f"  🔴 모든 포지션 청산")
        print(f"{'='*80}")
        
        current_price = self.get_current_price()
        closed = 0
        
        for position in positions:
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            close_price = current_price['bid'] if close_type == mt5.ORDER_TYPE_SELL else current_price['ask']
            
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.config['symbol'],
                "volume": position.volume,
                "type": close_type,
                "position": position.ticket,
                "price": close_price,
                "deviation": self.config['deviation'],
                "magic": self.config['magic_number'],
                "comment": "MANUAL_ALL",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(close_request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1
            time.sleep(0.05)
        
        print(f"\n✅ {closed}개 포지션 청산 완료!")
    
    def display_stats(self):
        """통계"""
        runtime = (datetime.now() - self.stats['start_time']).total_seconds() / 3600
        
        positions = mt5.positions_get(symbol=self.config['symbol'], magic=self.config['magic_number'])
        analysis = self.analyze_positions()
        
        # 현재 계좌 정보
        account_info = mt5.account_info()
        if account_info:
            current_balance = account_info.balance
            current_equity = account_info.equity
            balance_change = current_balance - self.stats['start_balance']
            equity_change = current_equity - self.stats['start_equity']
        else:
            current_balance = 0
            current_equity = 0
            balance_change = 0
            equity_change = 0
        
        print(f"\n{'='*80}")
        print(f"  📊 AI 그리드 봇 통계")
        print(f"{'='*80}")
        print(f"운영: {int(runtime)}시간 {int((runtime % 1) * 60)}분")
        print(f"")
        print(f"💰 계좌 현황:")
        print(f"  시작 잔고: ${self.stats['start_balance']:,.2f}")
        print(f"  현재 잔고: ${current_balance:,.2f}")
        print(f"  실제 수익: ${balance_change:+,.2f}")
        print(f"  현재 증거금: ${current_equity:,.2f} ({equity_change:+,.2f})")
        print(f"")
        print(f"📊 포지션: {len(positions) if positions else 0}개")
        print(f"  💙 수익: {len(analysis['profit_positions'])}개 (${analysis['total_profit']:+,.4f})")
        print(f"  ❤️ 손실: {len(analysis['loss_positions'])}개 (${analysis['total_loss']:+,.4f})")
        print(f"")
        print(f"📈 거래 통계:")
        print(f"  히트: {self.stats['grid_hits']} | 완료: {self.stats['total_trades']}")
        print(f"  🔄 방향전환: {self.stats['flips']}회")
        print(f"  ✅ 회피 손실: ${self.stats['avoided_loss']:,.2f}")
        
        if self.last_prediction:
            print(f"")
            print(f"🔮 최근 예측: {self.last_prediction['direction'].upper()} "
                  f"(신뢰도 {self.last_prediction['confidence']:.2%})")
        
        print(f"{'='*80}\n")
    
    def update_dashboard_data(self):
        """대시보드 데이터 업데이트"""
        current_price = self.get_current_price()
        if current_price:
            self.dashboard_data['prices'].append({
                'time': datetime.now().isoformat(),
                'price': current_price['ask']
            })
        
        self.dashboard_data['profits'].append({
            'time': datetime.now().isoformat(),
            'profit': self.stats['total_profit']
        })
    
    def get_dashboard_data(self):
        """대시보드 데이터 반환 (웹 서버용)"""
        analysis = self.analyze_positions()
        
        # 현재 가격 추가
        current_price = self.get_current_price()
        if current_price:
            # 가격 데이터가 비어있으면 초기화
            if len(self.dashboard_data['prices']) == 0:
                self.dashboard_data['prices'].append({
                    'time': datetime.now().isoformat(),
                    'price': current_price['ask']
                })
            
            # 수익 데이터가 비어있으면 초기화
            if len(self.dashboard_data['profits']) == 0:
                self.dashboard_data['profits'].append({
                    'time': datetime.now().isoformat(),
                    'profit': self.stats['total_profit']
                })
        
        return {
            'stats': {
                'total_profit': self.stats['total_profit'],
                'total_trades': self.stats['total_trades'],
                'grid_hits': self.stats['grid_hits'],
                'flips': self.stats['flips'],
                'avoided_loss': self.stats['avoided_loss'],
                'active_positions': len(self.active_positions),
                'profit_positions': len(analysis['profit_positions']),
                'loss_positions': len(analysis['loss_positions']),
            },
            'prices': list(self.dashboard_data['prices']),
            'profits': list(self.dashboard_data['profits']),
            'grid_changes': list(self.dashboard_data['grid_changes']),
            'flips': list(self.dashboard_data['flips']),
            'predictions': list(self.dashboard_data['predictions']),
            'last_prediction': self.last_prediction,
        }
    
    def keyboard_listener(self):
        """키보드 입력 감지"""
        print("\n" + "="*80)
        print("  ⌨️  키보드 명령")
        print("="*80)
        print("  H = 수익 포지션만 청산하고 종료 (파란불 💙)")
        print("  L = 손실 포지션만 청산하고 종료 (빨간불 ❤️)")
        print("  Q = 모든 포지션 청산하고 종료")
        print("  S = 현재 통계 보기")
        print("  C = 계속 실행")
        print("="*80 + "\n")
        
        while self.running:
            if msvcrt.kbhit():
                try:
                    key = msvcrt.getch()
                    
                    # bytes를 문자열로 변환
                    if isinstance(key, bytes):
                        key = key.decode('utf-8', errors='ignore').upper()
                    else:
                        key = str(key).upper()
                    
                    if key == 'H':
                        print("\n💙 수익 포지션 청산 명령 수신...")
                        self.manual_action = 'close_profit'
                        self.running = False
                        break
                    elif key == 'L':
                        print("\n❤️ 손실 포지션 청산 명령 수신...")
                        self.manual_action = 'close_loss'
                        self.running = False
                        break
                    elif key == 'Q':
                        print("\n🔴 모든 포지션 청산 명령 수신...")
                        self.manual_action = 'close_all'
                        self.running = False
                        break
                    elif key == 'S':
                        self.display_stats()
                    elif key == 'C':
                        print("\n▶️ 계속 실행 중...\n")
                except Exception as e:
                    pass  # 키 입력 에러 무시
            
            time.sleep(0.05)
    
    def run(self):
        """메인 루프"""
        # 키보드 리스너 시작
        listener_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        listener_thread.start()
        
        last_stats = time.time()
        last_prediction = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                
                # 현재 가격 수집 (ML 학습용)
                current_price = self.get_current_price()
                if current_price:
                    self.predictor.add_price(current_price['ask'], datetime.now())
                
                # 모델 재학습 (5분마다)
                if current_time - self.last_retrain >= self.config['ml_retrain_interval']:
                    print("\n🧠 모델 재학습 중...")
                    if self.predictor.train():
                        print("✓ 재학습 완료!")
                    self.last_retrain = current_time
                
                # 1분마다 예측 및 그리드 조정
                if current_time - last_prediction >= 60:
                    self.predict_and_adjust_grid()
                    last_prediction = current_time
                
                # 포지션 관리 (종료 명령이 없을 때만)
                if self.running:
                    self.check_and_manage_positions()
                
                # 대시보드 데이터 업데이트
                self.update_dashboard_data()
                
                # 통계 (30초마다)
                if current_time - last_stats >= 30:
                    self.display_stats()
                    last_stats = current_time
                
                # 실시간 표시
                if current_price:
                    positions = mt5.positions_get(symbol=self.config['symbol'], magic=self.config['magic_number'])
                    analysis = self.analyze_positions()
                    
                    # 현재 잔고
                    account_info = mt5.account_info()
                    if account_info:
                        balance_change = account_info.balance - self.stats['start_balance']
                        balance_str = f"잔고: ${balance_change:+,.2f}"
                    else:
                        balance_str = ""
                    
                    pred_str = ""
                    if self.last_prediction:
                        pred_str = f"🔮{self.last_prediction['direction'][0].upper()} "
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"{pred_str}BTC: ${current_price['ask']:,.2f} | "
                          f"💙{len(analysis['profit_positions'])} "
                          f"❤️{len(analysis['loss_positions'])} | "
                          f"{balance_str}", end='\r')
                
                time.sleep(self.config['check_interval'])
            
            # 루프 종료 후 수동 명령 처리
            print("\n\n종료 처리 중...")
            
            if self.manual_action == 'close_profit':
                self.close_profit_positions()
            elif self.manual_action == 'close_loss':
                self.close_loss_positions()
            elif self.manual_action == 'close_all':
                self.close_all_positions()
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Ctrl+C 감지 - 모든 포지션 청산 중...")
            
            # 모든 포지션 청산
            positions = mt5.positions_get(symbol=self.config['symbol'], magic=self.config['magic_number'])
            if positions:
                current_price = self.get_current_price()
                if current_price:
                    closed = 0
                    for position in positions:
                        close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                        close_price = current_price['bid'] if close_type == mt5.ORDER_TYPE_SELL else current_price['ask']
                        
                        close_request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": self.config['symbol'],
                            "volume": position.volume,
                            "type": close_type,
                            "position": position.ticket,
                            "price": close_price,
                            "deviation": self.config['deviation'],
                            "magic": self.config['magic_number'],
                            "comment": "CTRL_C_EXIT",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        
                        result = mt5.order_send(close_request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            closed += 1
                        time.sleep(0.05)
                    
                    print(f"✅ {closed}개 포지션 청산 완료!")
        
        finally:
            # 최종 계좌 정보 조회
            final_account = mt5.account_info()
            
            # 최종 통계
            self.display_stats()
            
            # 대기 주문 정리
            print("\n대기 주문 정리 중...")
            self.cancel_all_pending_orders()
            
            # 실제 수익 계산
            if final_account:
                final_balance = final_account.balance
                final_equity = final_account.equity
                actual_profit = final_balance - self.stats['start_balance']
                equity_change = final_equity - self.stats['start_equity']
            else:
                actual_profit = 0
                equity_change = 0
                final_balance = 0
                final_equity = 0
            
            print(f"\n{'='*80}")
            print(f"  🏁 봇 종료 - 최종 결산")
            print(f"{'='*80}")
            print(f"시작 잔고: ${self.stats['start_balance']:,.2f}")
            print(f"최종 잔고: ${final_balance:,.2f}")
            print(f"실제 수익: ${actual_profit:+,.2f}")
            print(f"")
            print(f"시작 증거금: ${self.stats['start_equity']:,.2f}")
            print(f"최종 증거금: ${final_equity:,.2f}")
            print(f"증거금 변화: ${equity_change:+,.2f}")
            print(f"")
            print(f"거래 통계:")
            print(f"  총 거래: {self.stats['total_trades']}회")
            print(f"  그리드 히트: {self.stats['grid_hits']}회")
            print(f"  방향 전환: {self.stats['flips']}회")
            print(f"  회피 손실: ${self.stats['avoided_loss']:,.2f}")
            print(f"{'='*80}\n")
            
            mt5.shutdown()
            print("✓ MT5 연결 종료")

# 전역 봇 인스턴스 (웹 서버에서 접근)
bot_instance = None

def main():
    global bot_instance
    
    print("\n" + "="*80)
    print("  🤖 AI 예측 그리드 봇 - 머신러닝 + 실시간 대시보드")
    print("="*80)
    print("\n핵심 기능:")
    print("  ✅ 1분 후 가격 예측 (Random Forest)")
    print("  ✅ 예측 기반 그리드 자동 조정")
    print("  ✅ 실시간 웹 대시보드 (http://localhost:5000)")
    print("  ✅ 손실 방향전환")
    print("  ✅ H/L/Q/S 키 수동 청산")
    
    bot_instance = AIGridBot(GRID_CONFIG)
    
    if not bot_instance.connect_mt5():
        sys.exit(1)
    
    if not bot_instance.get_symbol_info():
        mt5.shutdown()
        sys.exit(1)
    
    # 기존 포지션/주문 정리
    if not bot_instance.clear_existing_positions_and_orders():
        mt5.shutdown()
        sys.exit(1)
    
    # 과거 데이터 수집 및 학습
    if not bot_instance.collect_historical_data():
        mt5.shutdown()
        sys.exit(1)
    
    answer = input("\n그리드를 시작하시겠습니까? (y/n): ")
    if answer.lower() != 'y':
        mt5.shutdown()
        sys.exit(0)
    
    if not bot_instance.setup_grid():
        mt5.shutdown()
        sys.exit(1)
    
    # 웹 서버 시작
    print("\n🌐 웹 대시보드 시작 중...")
    from web_dashboard import start_web_server, set_bot_instance
    
    # 봇 인스턴스를 웹 서버에 전달
    set_bot_instance(bot_instance)
    
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    
    print("✓ 웹 대시보드: http://localhost:5000")
    
    bot_instance.run()

if __name__ == "__main__":
    main()
