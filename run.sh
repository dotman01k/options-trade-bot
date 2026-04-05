#!/bin/zsh
cd ~/Downloads/options_trade_bot
source venv/bin/activate
pkill -f streamlit 2>/dev/null
streamlit run app.py
