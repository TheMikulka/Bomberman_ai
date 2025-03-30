Spustitelný skript je main.py, který zajišťuje běh aplikace.
Skript nepříjmá žádné argumenty.

Základní setup: 1 skutečný hráč a 3 Q-learning agenti. Možno upravit ve tříde RunningState.

Ovládání hráče: W - pohyb nahoru
                S - pohyb dolů
                A - pohyb doleva
                D - pohyb doprava
                SPACE - Položení bomby

```bash
    # Vytvoření virtualního prostředí
    python -m venv venv

    # Aktivace virtualního prostředí
    # Pro Windows
    venv\Scripts\activate
    # Pro Unix nebo MacOS
    source venv/bin/activate

    # Instalace balíčků z requirements.txt
    pip install -r requirements.txt
```