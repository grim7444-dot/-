SPORTS = {
    "soccer": {
        "espn_sport": "soccer",
        "aliases": ["football", "축구", "soccer", "풋볼"],
        "emoji": "⚽",
        "leagues": {
            "premier": {"id": "eng.1", "name": "Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
            "laliga":  {"id": "esp.1", "name": "La Liga 🇪🇸"},
            "bundesliga": {"id": "ger.1", "name": "Bundesliga 🇩🇪"},
            "seriea":  {"id": "ita.1", "name": "Serie A 🇮🇹"},
            "ligue1":  {"id": "fra.1", "name": "Ligue 1 🇫🇷"},
            "mls":     {"id": "usa.1", "name": "MLS 🇺🇸"},
            "kleague": {"id": "kor.1", "name": "K League 1 🇰🇷"},
            "ucl":     {"id": "UEFA.CHAMPIONS", "name": "UEFA Champions League 🏆"},
        },
        "default_league": "premier",
    },
    "baseball": {
        "espn_sport": "baseball",
        "aliases": ["야구", "baseball", "mlb"],
        "emoji": "⚾",
        "leagues": {
            "mlb": {"id": "mlb", "name": "MLB 🇺🇸"},
        },
        "default_league": "mlb",
    },
    "basketball": {
        "espn_sport": "basketball",
        "aliases": ["농구", "basketball", "nba"],
        "emoji": "🏀",
        "leagues": {
            "nba": {"id": "nba", "name": "NBA 🇺🇸"},
            "wnba": {"id": "wnba", "name": "WNBA 🇺🇸"},
        },
        "default_league": "nba",
    },
    "football": {
        "espn_sport": "football",
        "aliases": ["미식축구", "nfl", "americanfootball"],
        "emoji": "🏈",
        "leagues": {
            "nfl": {"id": "nfl", "name": "NFL 🇺🇸"},
            "ncaa": {"id": "college-football", "name": "NCAA Football 🎓"},
        },
        "default_league": "nfl",
    },
    "hockey": {
        "espn_sport": "hockey",
        "aliases": ["아이스하키", "hockey", "nhl", "하키"],
        "emoji": "🏒",
        "leagues": {
            "nhl": {"id": "nhl", "name": "NHL 🇺🇸"},
        },
        "default_league": "nhl",
    },
}

ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"

COLORS = {
    "win":     0x2ECC71,
    "loss":    0xE74C3C,
    "default": 0x3498DB,
    "warning": 0xF39C12,
}
