"""Game modules. REWARD10_GAMES_ROOT adds a game's own folder, so a new game needs no change here."""
import os

if os.environ.get('REWARD10_GAMES_ROOT'):
    __path__.insert(0, os.path.join(os.environ['REWARD10_GAMES_ROOT'], 'games'))
