"""
News Sentiment Module — fetch economic/political/crypto news and analyze sentiment.

Sources:
  1. CryptoPanic API (free) — crypto-specific news with community sentiment
  2. Alternative.me Fear & Greed — already integrated
  3. Macro news keywords from RSS feeds (Fed, CPI, tariffs, regulations)

The sentiment score is added as an ML feature and used for trade filtering.
"""
import logging
import requests
import re
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Free news sources
CRYPTOPANIC_API = "https://cryptopanic.com/api/free/v1/posts/"
NEWSDATA_API = "https://newsdata.io/api/1/news"

# Bearish keywords (macro + crypto)
# Poids des keywords: (keyword, poids)
BEARISH_KEYWORDS_WEIGHTED = [
    # CRITICAL (3.0)
    ("nuclear", 3.0), ("invasion", 3.0), ("collapse", 3.0), ("bank run", 3.0),
    ("stablecoin depeg", 3.0), ("depeg", 3.0), ("chapter 11", 3.0), ("contagion", 3.0),
    ("hack", 2.5), ("exploit", 2.5), ("rug pull", 3.0), ("ponzi", 2.5),
    # HIGH (2.0)
    ("rate hike", 2.0), ("hawkish", 2.0), ("recession", 2.0), ("stagflation", 2.0),
    ("sec lawsuit", 2.0), ("sec charges", 2.0), ("ban crypto", 2.0), ("ban bitcoin", 2.0),
    ("liquidat", 2.0), ("margin call", 2.0), ("forced selling", 2.0),
    ("crash", 2.0), ("crisis", 2.0), ("panic", 2.0),
    # MEDIUM (1.0)
    ("tightening", 1.0), ("higher for longer", 1.0), ("inflation rises", 1.0),
    ("war", 1.0), ("conflict escalat", 1.0), ("sanctions", 1.0), ("tariff", 1.0),
    ("crackdown", 1.0), ("restrict", 1.0), ("selloff", 1.0), ("sell-off", 1.0),
    ("dump", 1.0), ("outflow", 1.0), ("bear market", 1.0), ("fud", 1.0),
    ("fraud", 1.0), ("scam", 1.0), ("delisting", 1.0), ("insolvency", 1.0),
    ("default", 1.0), ("downgrade", 1.0), ("deficit", 1.0),
    # LOW (0.5)
    ("quantitative tightening", 0.5), ("no rate cut", 0.5), ("delay cut", 0.5),
    ("unemployment rises", 0.5), ("gdp contracts", 0.5), ("debt ceiling", 0.5),
    ("trade war", 0.5), ("geopolitical risk", 0.5), ("layoffs", 0.5),
    ("whale sell", 0.5), ("plunge", 0.5), ("tumble", 0.5),
]
BEARISH_KEYWORDS = [k for k, _ in BEARISH_KEYWORDS_WEIGHTED]


# Bullish keywords
BULLISH_KEYWORDS_WEIGHTED = [
    # CRITICAL (3.0)
    ("etf approved", 3.0), ("spot etf", 3.0), ("legal tender", 3.0),
    ("strategic reserve", 3.0), ("treasury buy", 3.0),
    # HIGH (2.0)
    ("rate cut", 2.0), ("dovish", 2.0), ("pivot", 2.0), ("quantitative easing", 2.0),
    ("all-time high", 2.0), ("ath", 2.0), ("breakout", 2.0), ("bull run", 2.0),
    ("institutional", 2.0), ("etf approval", 2.0), ("pro-crypto", 2.0),
    ("halving", 2.0), ("ceasefire", 2.0), ("peace deal", 2.0),
    # MEDIUM (1.0)
    ("soft landing", 1.0), ("inflation falls", 1.0), ("inflation cools", 1.0),
    ("gdp growth", 1.0), ("rally", 1.0), ("inflow", 1.0), ("accumulation", 1.0),
    ("whale buy", 1.0), ("bull market", 1.0), ("adoption", 1.0),
    ("partnership", 1.0), ("regulation clarity", 1.0), ("approved", 1.0),
    ("mainnet", 1.0), ("upgrade", 1.0), ("stimulus", 1.0),
    # LOW (0.5)
    ("easing", 0.5), ("lower rates", 0.5), ("pause hike", 0.5),
    ("trade deal", 0.5), ("sanctions lifted", 0.5), ("tariff removed", 0.5),
    ("consumer confidence", 0.5), ("jobs beat", 0.5), ("momentum", 0.5),
    ("ai breakthrough", 0.5), ("tech rally", 0.5), ("innovation", 0.5),
    ("launch", 0.5), ("integration", 0.5),
]
BULLISH_KEYWORDS = [k for k, _ in BULLISH_KEYWORDS_WEIGHTED]

# Pondération par source (fiabilité + réactivité)
SOURCE_TRUST = {
    "cryptopanic":      1.4,   # Crypto-spécifique + votes communauté
    "theblock":         1.3,   # Référence institutionnelle
    "coindesk":         1.2,   # Référence grand public
    "cointelegraph":    1.1,
    "bitcoinmagazine":  1.0,
    "decrypt":          1.0,
    "newsdata":         0.9,   # Macro, moins crypto-précis
    "bbc":              0.8,   # Général, peu crypto
    "rss":              0.8,   # Source inconnue = poids réduit
    "unknown":          0.7,
}



class NewsSentiment:
    def __init__(self, config):
        self.config = config
        self._cryptopanic_token = getattr(config, "CRYPTOPANIC_TOKEN", "")
        self._newsdata_key = getattr(config, "NEWSDATA_API_KEY", "")

        # Cache
        self._last_fetch: datetime | None = None
        self._cache_minutes: int = 30  # Fetch news every 30 min (was 3 — too aggressive, risks 429)
        self._cached_score: float = 0.0
        self._cached_headlines: list = []
        self._cached_summary: dict = {}
        self._db_path = getattr(config, 'DB_PATH', 'trading_data.db')
        self._init_db()
        self._all_headlines_history: list = []  # Full session history
        # Pre-compile word-boundary regex for keyword matching
        self._bull_re = re.compile(
            r'\b(' + '|'.join(re.escape(k) for k in BULLISH_KEYWORDS) + r')\b',
            re.IGNORECASE,
        )
        self._bear_re = re.compile(
            r'\b(' + '|'.join(re.escape(k) for k in BEARISH_KEYWORDS) + r')\b',
            re.IGNORECASE,
        )

    def _init_db(self):
        """Create news tables if they dont exist."""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_sentiment_history (
                    timestamp TEXT,
                    score REAL,
                    label TEXT,
                    headline_count INTEGER,
                    bullish_count INTEGER,
                    bearish_count INTEGER,
                    neutral_count INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_headlines (
                    timestamp TEXT,
                    title TEXT,
                    source TEXT,
                    sentiment TEXT
                )
            """)
            conn.commit()
            conn.close()
            logger.info("News DB tables ready")
        except Exception as e:
            logger.warning(f"News DB init failed: {e}")

    def _save_to_db(self, summary: dict, headlines: list):
        """Save sentiment and headlines to database."""
        try:
            conn = sqlite3.connect(self._db_path)
            now = datetime.now().isoformat()

            # Save sentiment score
            conn.execute(
                "INSERT INTO news_sentiment_history VALUES (?,?,?,?,?,?,?)",
                (now, summary["score"], summary["label"],
                 summary["headline_count"], summary["bullish_count"],
                 summary["bearish_count"], summary["neutral_count"])
            )

            # Save headlines (last 30 only to avoid bloat)
            for h in headlines[:30]:
                conn.execute(
                    "INSERT INTO news_headlines VALUES (?,?,?,?)",
                    (now, h.get("title","")[:200], h.get("source",""), h.get("sentiment","neutral"))
                )

            # Cleanup: keep only last 7 days
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            conn.execute("DELETE FROM news_sentiment_history WHERE timestamp < ?", (week_ago,))
            conn.execute("DELETE FROM news_headlines WHERE timestamp < ?", (week_ago,))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"News DB save failed: {e}")

    def get_sentiment(self) -> dict:
        """Get current news sentiment. Returns cached if recent."""
        now = datetime.now()
        if self._last_fetch and (now - self._last_fetch).total_seconds() < self._cache_minutes * 60:
            return self._cached_summary

        # Fetch from all sources
        headlines = []

        # Source 1: CryptoPanic
        cp_news = self._fetch_cryptopanic()
        if cp_news:
            headlines.extend(cp_news)

        # Source 2: NewsData.io (macro news)
        macro_news = self._fetch_macro_news()
        if macro_news:
            headlines.extend(macro_news)

        # Source 3: RSS fallback
        rss_news = self._fetch_rss_news()
        if rss_news:
            headlines.extend(rss_news)

        # Analyze sentiment
        if headlines:
            score = self._analyze_sentiment(headlines, symbol=getattr(self, '_current_symbol', None))
        else:
            score = 0.0  # Neutral if no news

        self._cached_score = score
        self._cached_headlines = headlines[-20:]  # Keep last 20
        self._last_fetch = now

        self._cached_summary = {
            "score": score,  # -1.0 (very bearish) to +1.0 (very bullish)
            "label": self._score_to_label(score),
            "headline_count": len(headlines),
            "bullish_count": sum(1 for h in headlines if h.get("sentiment") == "bullish"),
            "bearish_count": sum(1 for h in headlines if h.get("sentiment") == "bearish"),
            "neutral_count": sum(1 for h in headlines if h.get("sentiment") == "neutral"),
            "top_headlines": [h["title"] for h in headlines[:5]],
            "timestamp": now.isoformat(),
        }

        logger.info(
            f"News sentiment: {score:+.2f} ({self._cached_summary['label']}) — "
            f"{len(headlines)} headlines "
            f"({self._cached_summary['bullish_count']}↑ {self._cached_summary['bearish_count']}↓)"
        )

        # Save to DB
        self._save_to_db(self._cached_summary, headlines)

        # Keep full history in memory
        self._all_headlines_history.extend(headlines)
        self._all_headlines_history = self._all_headlines_history[-200:]

        return self._cached_summary

    def get_sentiment_for_symbol(self, symbol: str) -> dict:
        """Get sentiment avec filtrage par symbole."""
        self._current_symbol = symbol
        result = self.get_sentiment()
        self._current_symbol = None
        return result

    def get_score(self) -> float:
        """Get sentiment score as a single float for ML features (-1 to +1)."""
        summary = self.get_sentiment()
        return summary["score"]

    def get_score_normalized(self) -> float:
        """Get sentiment normalized to 0-100 (like Fear & Greed)."""
        return (self.get_score() + 1) * 50  # -1→0, 0→50, +1→100

    # ------------------------------------------------------------------
    # Data sources
    # ------------------------------------------------------------------

    def _fetch_cryptopanic(self) -> list:
        """Fetch from CryptoPanic API (free tier — no token needed for public)."""
        try:
            params = {
                "auth_token": self._cryptopanic_token,
                "public": "true",
                "kind": "news",
                "filter": "important",
            }
            if not self._cryptopanic_token:
                # Free public endpoint (limited)
                params.pop("auth_token")

            resp = requests.get(
                CRYPTOPANIC_API,
                params=params,
                timeout=10,
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            results = []
            for post in data.get("results", [])[:20]:
                title = post.get("title", "")
                votes = post.get("votes", {})
                positive = votes.get("positive", 0) + votes.get("liked", 0)
                negative = votes.get("negative", 0) + votes.get("disliked", 0)

                if positive > negative:
                    sent = "bullish"
                elif negative > positive:
                    sent = "bearish"
                else:
                    sent = "neutral"

                results.append({
                    "title": title,
                    "source": "cryptopanic",
                    "sentiment": sent,
                    "time": post.get("published_at", ""),
                    "votes_pos": positive,
                    "votes_neg": negative,
                })

            logger.debug(f"CryptoPanic: {len(results)} headlines")
            return results

        except Exception as e:
            logger.debug(f"CryptoPanic fetch failed: {e}")
            return []

    def _fetch_macro_news(self) -> list:
        """Fetch macro/economic news from NewsData.io (free tier: 200 req/day)."""
        if not self._newsdata_key:
            return []

        try:
            resp = requests.get(
                NEWSDATA_API,
                params={
                    "apikey": self._newsdata_key,
                    "q": "federal reserve OR inflation OR crypto regulation OR bitcoin",
                    "language": "en",
                    "category": "business,politics",
                },
                timeout=10,
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            results = []
            for article in data.get("results", [])[:15]:
                title = article.get("title", "")
                sent = self._keyword_sentiment(title)
                results.append({
                    "title": title,
                    "source": "newsdata",
                    "sentiment": sent,
                    "time": article.get("pubDate", ""),
                })

            logger.debug(f"NewsData: {len(results)} headlines")
            return results

        except Exception as e:
            logger.debug(f"NewsData fetch failed: {e}")
            return []

    def _fetch_rss_news(self) -> list:
        """Fetch from free RSS feeds as fallback."""
        feeds = [
            "https://cointelegraph.com/rss",
            "https://coindesk.com/arc/outboundfeeds/rss/",
            "https://www.theblock.co/rss.xml",
            "https://decrypt.co/feed",
            "https://bitcoinmagazine.com/.rss/full/",
            "https://feeds.bbci.co.uk/news/business/rss.xml",
            # Removed: https://feeds.reuters.com/reuters/businessNews — Reuters shut down public RSS in 2020
        ]
        results = []

        # Mapping URL → nom de source précis
        feed_names = {
            "cointelegraph.com":    "cointelegraph",
            "coindesk.com":         "coindesk",
            "theblock.co":          "theblock",
            "decrypt.co":           "decrypt",
            "bitcoinmagazine.com":  "bitcoinmagazine",
            "bbci.co.uk":           "bbc",
        }

        for feed_url in feeds:
            # Identifier la source par l'URL
            src_name = "rss"
            for domain, name in feed_names.items():
                if domain in feed_url:
                    src_name = name
                    break
            try:
                resp = requests.get(feed_url, timeout=8)
                if resp.status_code != 200:
                    continue

                # Extraire items complets (title + pubDate)
                items = re.findall(
                    r"<item>(.*?)</item>",
                    resp.text, re.DOTALL
                )
                if not items:
                    # Fallback: juste les titres
                    items_text = [resp.text]
                else:
                    items_text = items[:10]

                for item in items_text[:10]:
                    # Titre
                    tm = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", item)
                    if not tm:
                        continue
                    title = tm.group(1).strip()
                    if len(title) < 10:
                        continue

                    # pubDate
                    pm = re.search(r"<pubDate>(.*?)</pubDate>", item)
                    pub_date = pm.group(1).strip() if pm else ""

                    sent = self._keyword_sentiment(title)
                    results.append({
                        "title": title,
                        "source": src_name,
                        "sentiment": sent,
                        "time": pub_date,
                    })

            except Exception as e:
                logger.debug(f"RSS fetch failed {feed_url}: {e}")

        logger.debug(f"RSS: {len(results)} headlines")
        return results

    # ------------------------------------------------------------------
    # Sentiment analysis
    # ------------------------------------------------------------------

    def _keyword_sentiment(self, text: str) -> str:
        """Keyword-based sentiment analysis using word-boundary regex
        (avoids substring false positives like 'war' in 'award' or 'ban' in 'urban')."""
        if not text:
            return "neutral"
        bull_score = len(self._bull_re.findall(text))
        bear_score = len(self._bear_re.findall(text))

        if bull_score > bear_score:
            return "bullish"
        elif bear_score > bull_score:
            return "bearish"
        return "neutral"

    def _deduplicate(self, headlines: list) -> list:
        """Supprime les doublons par similarite de titre (mots communs > 60%)."""
        seen = []
        unique = []
        for h in headlines:
            title_words = set(h.get("title", "").lower().split())
            is_dup = False
            for s in seen:
                common = len(title_words & s) / max(len(title_words | s), 1)
                if common > 0.6:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(h)
                seen.append(title_words)
        return unique

    def _time_weight(self, time_str: str) -> float:
        """Poids temporel: news recente = plus important. Max 2.0, min 0.5."""
        try:
            from datetime import timezone
            import dateutil.parser
            pub = dateutil.parser.parse(time_str)
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            age_h = (datetime.now(timezone.utc) - pub).total_seconds() / 3600
            if age_h < 1:   return 2.0
            if age_h < 3:   return 1.5
            if age_h < 6:   return 1.0
            if age_h < 12:  return 0.75
            return 0.5
        except Exception:
            return 1.0

    def _symbol_relevance(self, title: str, symbol: str) -> float:
        """Boost si la news mentionne explicitement le symbole."""
        sym_map = {
            "SOL/USDT": ["sol", "solana"],
            "BTC/USDT": ["btc", "bitcoin"],
            "ETH/USDT": ["eth", "ethereum"],
        }
        keywords = sym_map.get(symbol, [])
        title_lower = title.lower()
        for kw in keywords:
            if kw in title_lower:
                return 1.5
        return 1.0

    def _analyze_sentiment(self, headlines: list, symbol: str = None) -> float:
        """Score de sentiment ameliore avec poids, temps, dedup et consensus."""
        # E — Deduplication
        headlines = self._deduplicate(headlines)

        bull_score = 0.0
        bear_score = 0.0
        source_scores = {}

        # Construire dict de poids
        bull_weights = {k: w for k, w in BULLISH_KEYWORDS_WEIGHTED}
        bear_weights = {k: w for k, w in BEARISH_KEYWORDS_WEIGHTED}

        for h in headlines:
            title = h.get("title", "").lower()
            source = h.get("source", "unknown")
            time_str = h.get("time", "")
            tw = self._time_weight(time_str)
            sr = self._symbol_relevance(title, symbol) if symbol else 1.0

            # A — Poids par keyword avec détection négation
            def _negated(text, keyword):
                idx = text.find(keyword)
                if idx < 0:
                    return False
                window = text[max(0, idx-30):idx]
                return any(neg in window for neg in ["no ", "not ", "without ", "lifted", "removed", "prevented", "denied", "false", "avoid"])

            h_bull = sum(w for k, w in bull_weights.items() if k in title and not _negated(title, k))
            h_bear = sum(w for k, w in bear_weights.items() if k in title and not _negated(title, k))
            # Inversion si négation détectée (ex: "ban lifted" → bullish)
            bull_neg = sum(w for k, w in bull_weights.items() if k in title and _negated(title, k))
            bear_neg = sum(w for k, w in bear_weights.items() if k in title and _negated(title, k))
            h_bull += bear_neg * 0.5   # "hack prevented" → léger bullish
            h_bear += bull_neg * 0.5   # "etf rejected" → léger bearish

            # D — Normalisation votes CryptoPanic
            if source == "cryptopanic":
                pos = h.get("votes_pos", 0)
                neg = h.get("votes_neg", 0)
                total_votes = pos + neg
                if total_votes >= 5:
                    vote_bias = (pos - neg) / total_votes  # -1 a +1
                    h_bull *= (1 + max(0, vote_bias) * 0.5)
                    h_bear *= (1 + max(0, -vote_bias) * 0.5)

            # Pondération par source
            trust = SOURCE_TRUST.get(source, 0.8)
            h_bull *= tw * sr * trust
            h_bear *= tw * sr * trust

            bull_score += h_bull
            bear_score += h_bear

            # Tracker par source pour consensus
            src_net = h_bull - h_bear
            source_scores[source] = source_scores.get(source, 0) + src_net

        total = bull_score + bear_score
        if total == 0:
            return 0.0

        raw_score = (bull_score - bear_score) / total

        # F — Score de consensus multi-sources
        if len(source_scores) >= 2:
            signs = [1 if v > 0 else -1 if v < 0 else 0 for v in source_scores.values()]
            consensus = sum(signs) / len(signs)  # -1 a +1
            # Si consensus fort dans meme sens, amplifier legerement
            raw_score = raw_score * (1 + 0.2 * abs(consensus))
            raw_score = max(-1.0, min(1.0, raw_score))

        return round(raw_score, 4)


    def _score_to_label(self, score: float) -> str:
        if score >= 0.3:
            return "Bullish"
        elif score >= 0.1:
            return "Slightly Bullish"
        elif score <= -0.3:
            return "Bearish"
        elif score <= -0.1:
            return "Slightly Bearish"
        return "Neutral"

    # ------------------------------------------------------------------
    # For dashboard
    # ------------------------------------------------------------------

    def get_headlines(self) -> list:
        """Return cached headlines for dashboard display."""
        return self._cached_headlines

    def get_summary(self) -> dict:
        """Return full summary for API/dashboard."""
        if not self._cached_summary:
            self.get_sentiment()
        return self._cached_summary


    def get_sentiment_history(self, hours: int = 24) -> list:
        """Load sentiment history from DB."""
        try:
            conn = sqlite3.connect(self._db_path)
            cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
            rows = conn.execute(
                "SELECT * FROM news_sentiment_history WHERE timestamp > ? ORDER BY timestamp ASC",
                (cutoff,)
            ).fetchall()
            conn.close()
            return [{"timestamp": r[0], "score": r[1], "label": r[2],
                     "headlines": r[3], "bullish": r[4], "bearish": r[5], "neutral": r[6]}
                    for r in rows]
        except Exception:
            return []

    def get_headlines_history(self, hours: int = 24, limit: int = 100) -> list:
        """Load headlines from DB."""
        try:
            conn = sqlite3.connect(self._db_path)
            cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
            rows = conn.execute(
                "SELECT * FROM news_headlines WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?",
                (cutoff, limit)
            ).fetchall()
            conn.close()
            return [{"timestamp": r[0], "title": r[1], "source": r[2], "sentiment": r[3]}
                    for r in rows]
        except Exception:
            return []
