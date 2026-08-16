from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
import datetime
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    name = db.Column(db.String(150), nullable=False)
    
    # Financial settings
    electricity_rate = db.Column(db.Float, default=225.0) # e.g. NGN/kWh
    currency = db.Column(db.String(10), default="NGN")
    
    # Preferences stored as JSON
    live_settings = db.Column(db.Text, default='{}')
    appliances = db.Column(db.Text, default='[]')
    
    # Relationships
    badges = db.relationship('UserBadge', backref='user', lazy=True)
    logs = db.relationship('DailyLog', backref='user', lazy=True)

class UserBadge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    badge_name = db.Column(db.String(100), nullable=False)
    date_earned = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class DailyLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date_str = db.Column(db.String(10), nullable=False) # YYYY-MM-DD
    grid_import_kWh = db.Column(db.Float, default=0.0)
    solar_used_kWh = db.Column(db.Float, default=0.0)
    money_saved = db.Column(db.Float, default=0.0)
