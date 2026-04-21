"""HUNTER Configuration -- Domains, constants, scoring thresholds."""

# === Model Configuration — three-tier routing ===
# DEEP:  heaviest reasoning — hypothesis formation, mechanism kill, scoring
# MODEL: standard — other kill rounds, refinement, entity resolution, chain extension
# FAST:  volume — ingest extraction, anomaly detection, gap queries, synonym expansion
MODEL_DEEP = "claude-opus-4-7"       # best reasoning, ~5x Sonnet cost; used sparingly
MODEL = "claude-sonnet-4-5"          # standard; mid-tier reasoning at moderate cost
MODEL_FAST = "claude-haiku-4-5"      # volume; fast + cheap for repetitive extraction
MAX_TOKENS_RESPONSE = 4096
MAX_TOKENS_RESPONSE_SEARCH = 2048
MAX_TOKENS_RESPONSE_REPORT = 8192
MAX_TOKENS_RESPONSE_DEEP_DIVE = 8192

# === Token Caps (output tokens only -- web search inflates input massively) ===
TOKEN_CAP_NORMAL = 50_000
TOKEN_CAP_DEEP_DIVE = 150_000

# === Timeout (seconds) ===
TIMEOUT_NORMAL = 30 * 60       # 30 minutes
TIMEOUT_DEEP_DIVE = 3 * 60 * 60  # 3 hours

# === Diamond Scale Thresholds ===
THRESHOLDS = {
    "noise":       (1, 19),
    "interesting": (20, 39),
    "notable":     (40, 59),
    "strong":      (60, 74),
    "diamond":     (75, 89),
    "legendary":   (90, 100),
}

DEEP_DIVE_THRESHOLD = 75
REPORT_THRESHOLD = 40

# === Cross-Domain Ranking Adjustments ===
ACTIONABILITY_RANGE = (0.7, 1.3)
CONFIDENCE_PENALTY_RANGE = (0, -15)
PERSONAL_FIT_BONUS_RANGE = (0, 10)

# === Database ===
DB_PATH = "hunter.db"

# === Dashboard ===
DASHBOARD_REFRESH_SECONDS = 30

# === The 14 Hunting Domains ===
DOMAINS = [
    {
        "id": 1,
        "name": "Mathematics",
        "description": "Unsolved problems where recent computational advances create new approaches",
        "search_strategy": "Search for recent mathematical breakthroughs, unsolved problems with new computational approaches, Fields Medal candidates, arxiv preprints in applied mathematics",
        "diamond_signal": "A Millennium Prize problem where a solution approach using modern computation is being overlooked",
        "example_searches": [
            "recent breakthroughs applied mathematics 2025 2026",
            "unsolved math problems computational approaches",
            "mathematical techniques with commercial applications",
        ],
    },
    {
        "id": 2,
        "name": "Economics",
        "description": "Market inefficiencies, policy gaps, untested economic theories with practical implications",
        "search_strategy": "Search for new economic regulations, policy changes, trade imbalances, emerging market inefficiencies, untested economic models",
        "diamond_signal": "A new EU regulation that creates an arbitrage between two markets nobody has connected",
        "example_searches": [
            "new economic regulations 2025 2026 implications",
            "market inefficiency emerging economies",
            "economic policy gaps opportunities",
        ],
    },
    {
        "id": 3,
        "name": "Technology / AI",
        "description": "Capabilities nobody is building yet, research breakthroughs with near-term applications",
        "search_strategy": "Search for new AI capabilities, research papers with practical implications, technology gaps, tools nobody has built yet",
        "diamond_signal": "A new model capability that has immediate real-time application but nobody is building the implementation",
        "example_searches": [
            "AI research breakthroughs practical applications 2025 2026",
            "technology gaps nobody building",
            "new AI capabilities underutilised",
        ],
    },
    {
        "id": 4,
        "name": "Finance / Investing",
        "description": "Undervalued assets, upcoming catalysts, sectors with contrarian opportunity",
        "search_strategy": "Search for undervalued sectors, upcoming catalysts, contrarian investment theses, macro trends creating pricing inefficiencies",
        "diamond_signal": "A sector where consensus is wrong and a specific contradiction in market pricing reflects a mispricing",
        "example_searches": [
            "undervalued sectors 2025 2026 contrarian",
            "upcoming market catalysts overlooked",
            "pricing inefficiency financial markets",
        ],
    },
    {
        "id": 5,
        "name": "Betting / Prediction",
        "description": "Mispriced odds, statistical models that outperform bookmakers, prediction market inefficiencies",
        "search_strategy": "Search for prediction market mispricing, sports analytics models, political betting odds vs fundamentals, statistical arbitrage opportunities",
        "diamond_signal": "A championship match where public bias has created consistently exploitable odds",
        "example_searches": [
            "prediction market mispricing 2025 2026",
            "sports betting model statistical edge",
            "political prediction markets inefficiency",
        ],
    },
    {
        "id": 6,
        "name": "Products / SaaS",
        "description": "Gaps in the market, things people complain about that nobody has built a proper solution for",
        "search_strategy": "Search for common complaints about existing tools, product gaps, underserved niches, industries still using spreadsheets or manual processes",
        "diamond_signal": "An industry where the leading software exists but everyone complains about it",
        "example_searches": [
            "software complaints industries manual processes 2025",
            "product gaps underserved markets SaaS",
            "industries need better tools automation",
        ],
    },
    {
        "id": 7,
        "name": "Geopolitics",
        "description": "Upcoming elections, trade policy shifts, sanctions creating opportunities",
        "search_strategy": "Search for upcoming policy changes, trade shifts, diplomatic realignments, sanctions creating investment or trade opportunities",
        "diamond_signal": "A predicted diplomatic or trade realignment that creates economic opportunities nobody is acting on",
        "example_searches": [
            "upcoming trade policy changes 2025 2026",
            "geopolitical shifts economic implications",
            "sanctions creating opportunities",
        ],
    },
    {
        "id": 8,
        "name": "Science / Research",
        "description": "Breakthroughs with commercial potential, patents expiring, research nearing application",
        "search_strategy": "Search for recent scientific publications with commercial potential, research breakthroughs nearing productisation, university spin-outs",
        "diamond_signal": "A newly published research paper that demonstrates something with immediate commercial potential",
        "example_searches": [
            "scientific breakthroughs commercial potential 2025 2026",
            "research nearing product application",
            "university research spin-out opportunities",
        ],
    },
    {
        "id": 9,
        "name": "Banking / FinTech",
        "description": "Regulatory changes, new licensing frameworks, compliance gaps creating product opportunities",
        "search_strategy": "Search for banking regulation changes, new fintech licensing, open banking developments, compliance requirements creating product needs",
        "diamond_signal": "A country that just passed new banking regulation creating a specific product or platform opportunity",
        "example_searches": [
            "banking regulation changes 2025 2026",
            "fintech licensing new markets",
            "open banking compliance gaps opportunities",
        ],
    },
    {
        "id": 10,
        "name": "Content / Media",
        "description": "Underserved content niches, platform algorithm changes creating opportunity, emerging formats",
        "search_strategy": "Search for platform algorithm changes, underserved content niches, new social media features creating content opportunities, creator economy gaps",
        "diamond_signal": "A new social platform feature or algorithmic change that creates a first-mover content opportunity",
        "example_searches": [
            "social media algorithm changes 2025 2026",
            "underserved content niches creator economy",
            "platform features new content opportunities",
        ],
    },
    {
        "id": 11,
        "name": "Gaming",
        "description": "Underserved game genres, emerging platforms, player community gaps with zero competition",
        "search_strategy": "Search for underserved gaming genres, emerging gaming platforms, gaps in gaming communities, new gaming technologies without products",
        "diamond_signal": "A classic game genre with a massive audience but zero modern competition",
        "example_searches": [
            "underserved game genres 2025",
            "emerging gaming platform opportunities",
            "gaming community gaps no competition",
        ],
    },
    {
        "id": 12,
        "name": "Real Estate",
        "description": "Regulatory arbitrage, undervalued markets, policy changes creating windows of opportunity",
        "search_strategy": "Search for property market regulatory changes, undervalued real estate markets, zoning changes, emerging real estate technologies",
        "diamond_signal": "A policy or zoning change that opens up a specific real estate opportunity in an undervalued area",
        "example_searches": [
            "real estate regulatory changes 2025 2026",
            "undervalued property markets opportunity",
            "zoning changes real estate opportunities",
        ],
    },
    {
        "id": 13,
        "name": "Legal / IP",
        "description": "Expiring patents with commercial potential, trademark gaps, legal framework changes creating opportunities",
        "search_strategy": "Search for expiring patents, trademark gaps, legal framework changes, IP licensing opportunities, regulatory shifts creating new business categories",
        "diamond_signal": "A pharma or tech patent expiring within 6 months that creates a business or licensing opportunity",
        "example_searches": [
            "expiring patents commercial potential 2025 2026",
            "trademark gaps business opportunities",
            "legal framework changes new business categories",
        ],
    },
    {
        "id": 14,
        "name": "History / Knowledge",
        "description": "Historical patterns that map to current situations, forgotten innovations worth reviving",
        "search_strategy": "Search for historical parallels to current situations, forgotten technologies or innovations, cyclical patterns, historical case studies with modern applications",
        "diamond_signal": "A forgotten historical technology or business model that modern conditions make viable again",
        "example_searches": [
            "historical patterns current economic situation",
            "forgotten innovations worth reviving",
            "historical parallels modern technology",
        ],
    },
]

# Domain name lookup
DOMAIN_NAMES = {d["id"]: d["name"] for d in DOMAINS}
DOMAIN_BY_NAME = {d["name"]: d for d in DOMAINS}

# === v2 Fact-Collision Architecture ===

# Ingest/Collision ratio -- 50/50 steady state
# Core source types approaching parity, balanced ingest and collision
INGEST_RATIO = 0.5

# Collision lookback windows
COLLISION_LOOKBACK_ANOMALIES = 7   # days
COLLISION_LOOKBACK_FACTS = 30      # days

# Kill phase
KILL_SEARCH_COUNT = 4  # 4 kill types: fact_check, mechanism, competitor, barrier

# Quality control
HYPOTHESIS_RATE_CAP = 0.20  # max 20% of collisions should produce hypotheses

# Data source search queries for INGEST mode
# DESIGN PRINCIPLE: Each source type must pull from GENUINELY DIFFERENT professional
# worlds. A pharma patent and a pharma SEC filing are the same silo with different
# labels. True cross-domain collisions require facts from separate publication ecosystems.
DATA_SOURCES = [
    # === ANALYST_TARGET — Inverse HUNTER fuel. Published numbers to decompose. ===
    {"type": "analyst_target", "query": "Goldman Sachs price target upgrade downgrade equity 2026"},
    {"type": "analyst_target", "query": "Morgan Stanley price target revision industrials 2026"},
    {"type": "analyst_target", "query": "analyst consensus estimate EPS revision downward 2026"},
    {"type": "analyst_target", "query": "sell side research price target raised lowered 2026"},
    {"type": "analyst_target", "query": "credit rating agency outlook negative investment grade 2026"},
    {"type": "analyst_target", "query": "cap rate assumption REIT appraisal industry standard 2026"},
    {"type": "analyst_target", "query": "options implied volatility quarterly earnings 2026"},
    {"type": "analyst_target", "query": "insurance reserves actuarial assumption mortality table 2026"},
    {"type": "analyst_target", "query": "DCF discount rate WACC assumption analyst report 2026"},
    {"type": "analyst_target", "query": "CMBS rating agency loss severity assumption 2026"},

    # === PATENT — diversified across industries, not just pharma ===
    {"type": "patent", "query": "patent expired this month industrial manufacturing process 2026"},
    {"type": "patent", "query": "patent granted water treatment membrane filtration industrial 2026"},
    {"type": "patent", "query": "patent granted battery cathode anode novel material 2026"},
    {"type": "patent", "query": "patent granted C22B rare earth metal processing method 2026"},
    {"type": "patent", "query": "patent expiration semiconductor fabrication process 2026"},
    {"type": "patent", "query": "patent licensing agreement construction materials 2026"},
    {"type": "patent", "query": "PTAB inter partes review institution decision 2026"},
    {"type": "patent", "query": "patent expiration agricultural chemical herbicide pesticide 2026"},
    {"type": "patent", "query": "patent granted HVAC refrigerant compressor efficiency 2026"},
    {"type": "patent", "query": "patent expiration telecom wireless infrastructure 5G 2026"},
    # === BANKRUPTCY — across industries ===
    {"type": "bankruptcy", "query": "retail chain bankruptcy Chapter 11 store closures 2026"},
    {"type": "bankruptcy", "query": "construction company bankruptcy surety bond claim 2026"},
    {"type": "bankruptcy", "query": "trucking logistics company bankruptcy fleet auction 2026"},
    {"type": "bankruptcy", "query": "restaurant chain bankruptcy franchise agreement 2026"},
    {"type": "bankruptcy", "query": "Chapter 11 debtor-in-possession financing approved 2026"},
    {"type": "bankruptcy", "query": "bankruptcy 363 sale approved court order manufacturing 2026"},
    {"type": "bankruptcy", "query": "commercial real estate owner bankruptcy CMBS default 2026"},
    {"type": "bankruptcy", "query": "technology startup bankruptcy intellectual property auction 2026"},
    {"type": "bankruptcy", "query": "energy company bankruptcy power purchase agreement rejection 2026"},
    # === PHARMACEUTICAL — keep pharma-specific here, not elsewhere ===
    {"type": "pharmaceutical", "query": "pharmaceutical FDA approval this month"},
    {"type": "pharmaceutical", "query": "drug patent expiration generic competition 2026"},
    {"type": "pharmaceutical", "query": "FDA 505(b)(2) approval NDA new molecular entity 2026"},
    {"type": "pharmaceutical", "query": "FDA breakthrough therapy designation granted 2026"},
    {"type": "pharmaceutical", "query": "orphan drug designation rare disease FDA 2026"},
    {"type": "pharmaceutical", "query": "FDA Orange Book Paragraph IV certifications pending ANDA 2026"},
    {"type": "pharmaceutical", "query": "generic drug ANDA filed unapproved 2026 exclusivity expiry"},
    # === ACADEMIC — diversified beyond clinical trials ===
    {"type": "academic", "query": "academic paper retracted fraud investigation 2026"},
    {"type": "academic", "query": "arxiv preprint materials science novel catalyst 2026"},
    {"type": "academic", "query": "university technology transfer license granted engineering 2026"},
    {"type": "academic", "query": "climate science paper ocean current circulation model 2026"},
    {"type": "academic", "query": "structural engineering failure analysis building collapse study 2026"},
    {"type": "academic", "query": "agricultural science crop yield soil microbiome research 2026"},
    {"type": "academic", "query": "actuarial science insurance loss model methodology paper 2026"},
    {"type": "academic", "query": "urban planning transportation infrastructure demand study 2026"},
    # === SEC FILING — diversified beyond pharma 10-Ks ===
    {"type": "sec_filing", "query": "SEC 8-K material event filing real estate REIT 2026"},
    {"type": "sec_filing", "query": "IPO filed S-1 technology infrastructure 2026"},
    {"type": "sec_filing", "query": "SEC Schedule 13D activist stake acquired 2026"},
    {"type": "sec_filing", "query": "SEC Form 4 insider buying cluster small cap industrial 2026"},
    {"type": "sec_filing", "query": "SEC Form 12b-25 late filing notification 2026"},
    {"type": "sec_filing", "query": "REIT 10-K occupancy rate decline tenant concentration risk 2026"},
    {"type": "sec_filing", "query": "insurance company statutory filing surplus change 2026"},
    {"type": "sec_filing", "query": "13F filing new position unusual stake energy utility 2026"},
    {"type": "sec_filing", "query": "merger acquisition announced infrastructure logistics 2026"},
    # === REGULATION — diversified beyond FDA/pharma ===
    {"type": "regulation", "query": "EU regulation published Official Journal environmental 2026"},
    {"type": "regulation", "query": "Federal Register final rule published this week"},
    {"type": "regulation", "query": "trade tariff change announced 2026"},
    {"type": "regulation", "query": "EPA final rule PFAS discharge limit compliance deadline 2026"},
    {"type": "regulation", "query": "REACH regulation substance restriction effective date 2026"},
    {"type": "regulation", "query": "FERC order transmission grid interconnection queue reform 2026"},
    {"type": "regulation", "query": "state building code change energy efficiency requirement 2026"},
    {"type": "regulation", "query": "CMS Medicare reimbursement rate change final rule 2026"},
    {"type": "regulation", "query": "DOT pipeline safety regulation inspection requirement 2026"},
    {"type": "regulation", "query": "state insurance commissioner rate filing order 2026"},
    {"type": "regulation", "query": "UK FCA enforcement notice published 2026"},
    {"type": "regulation", "query": "zoning variance approved mixed use development 2026"},
    # === GOVERNMENT CONTRACT — diversified ===
    {"type": "government_contract", "query": "government contract awarded infrastructure construction 2026"},
    {"type": "government_contract", "query": "defense contract sole source justification published 2026"},
    {"type": "government_contract", "query": "GSA schedule contract modification deobligation 2026"},
    {"type": "government_contract", "query": "SBIR STTR Phase II award clean energy 2026"},
    {"type": "government_contract", "query": "VA hospital construction contract awarded 2026"},
    {"type": "government_contract", "query": "DOE loan guarantee program application energy storage 2026"},
    {"type": "government_contract", "query": "state highway bridge repair contract awarded 2026"},
    # === COMMODITY — diversified beyond metals ===
    {"type": "commodity", "query": "commodity price unexpected movement this week"},
    {"type": "commodity", "query": "COMEX registered warehouse inventory withdrawal silver gold 2026"},
    {"type": "commodity", "query": "LME warehouse stock report aluminum copper draw 2026"},
    {"type": "commodity", "query": "lumber futures price construction demand 2026"},
    {"type": "commodity", "query": "cement concrete price increase supply shortage 2026"},
    {"type": "commodity", "query": "natural gas storage report injection withdrawal 2026"},
    {"type": "commodity", "query": "shipping container freight rate Baltic index movement 2026"},
    {"type": "commodity", "query": "agricultural commodity crop report USDA surprise 2026"},
    # === JOB LISTING — diversified ===
    {"type": "job_listing", "query": "company mass layoff announcement this week"},
    {"type": "job_listing", "query": "chief restructuring officer appointed 2026"},
    {"type": "job_listing", "query": "data center hiring surge electrical engineer 2026"},
    {"type": "job_listing", "query": "REIT property manager hiring expansion 2026"},
    {"type": "job_listing", "query": "insurance company actuary hiring surge 2026"},
    {"type": "job_listing", "query": "construction superintendent hiring shortage 2026"},
    {"type": "job_listing", "query": "utility company lineman electrician hiring crisis 2026"},
    # === EARNINGS — diversified beyond pharma ===
    {"type": "earnings", "query": "CEO resignation announced this month"},
    {"type": "earnings", "query": "REIT earnings FFO miss occupancy decline 2026"},
    {"type": "earnings", "query": "insurance company combined ratio deterioration earnings 2026"},
    {"type": "earnings", "query": "utility company earnings capex guidance increase 2026"},
    {"type": "earnings", "query": "construction company backlog decline margin pressure 2026"},
    {"type": "earnings", "query": "logistics company earnings freight volume decline 2026"},
    {"type": "earnings", "query": "regional bank earnings CRE loan loss provision increase 2026"},
    # === APP RANKING — keep minimal ===
    {"type": "app_ranking", "query": "app store top charts biggest mover this week"},
    {"type": "app_ranking", "query": "fintech app download surge banking alternative 2026"},
    # === OTHER — diversified ===
    {"type": "other", "query": "class action lawsuit filed environmental contamination 2026"},
    {"type": "other", "query": "supply chain disruption factory closure this month"},
    {"type": "other", "query": "dam bridge infrastructure failure inspection report 2026"},
    {"type": "other", "query": "wildfire flood insurance claim surge state 2026"},
    {"type": "other", "query": "major lease termination tenant departure commercial 2026"},
    # === GENERIC DELAY ALPHA queries (keep — proven thesis framework) ===
    {"type": "pharmaceutical", "query": "clinical trial results published phase 3 2026"},
    {"type": "regulation", "query": "Office of Generic Drugs ANDA review backlog 2025 2026 approval delays"},
    {"type": "sec_filing", "query": "pharmaceutical company 10-K single product revenue dependency patent expiring 2026"},
    # === CORE DOMAIN TARGETED QUERIES ===
    # Domain 1: Commercial Real Estate Distress & Credit
    {"type": "cre_credit", "query": "CMBS delinquency rate increase office retail 2026"},
    {"type": "cre_credit", "query": "commercial real estate loan maturity wall refinancing gap 2026"},
    {"type": "cre_credit", "query": "office to residential conversion approved zoning change 2026"},
    {"type": "cre_credit", "query": "regional bank CRE exposure concentration risk FDIC 2026"},
    {"type": "cre_credit", "query": "Trepp CMBS special servicing rate increase 2026"},
    {"type": "cre_credit", "query": "commercial mortgage default foreclosure filing 2026"},
    {"type": "cre_credit", "query": "FDIC quarterly banking profile commercial real estate losses 2025 2026"},
    # Domain 2: Specialty Real Estate Supply/Demand
    {"type": "specialty_re", "query": "industrial warehouse vacancy rate supply constrained market 2026"},
    {"type": "specialty_re", "query": "grocery anchored retail center acquisition cap rate 2026"},
    {"type": "specialty_re", "query": "medical outpatient building MOB development pipeline 2026"},
    {"type": "specialty_re", "query": "senior housing occupancy rate NIC data 2026"},
    {"type": "specialty_re", "query": "data center power capacity constraint permitting delay 2026"},
    {"type": "specialty_re", "query": "build to rent single family rental portfolio acquisition 2026"},
    {"type": "specialty_re", "query": "marina portfolio transaction waterfront property acquisition 2026"},
    # Domain 3: Insurance Market Structure
    {"type": "insurance", "query": "D&O insurance premium rate change 2026 renewal cycle"},
    {"type": "insurance", "query": "property casualty insurance combined ratio deterioration 2026"},
    {"type": "insurance", "query": "specialty insurance line pricing hard market 2026"},
    {"type": "insurance", "query": "AM Best rating action downgrade insurance company 2026"},
    {"type": "insurance", "query": "NAIC statutory filing insurance company surplus change 2026"},
    {"type": "insurance", "query": "insurance run-off transaction legacy liability acquisition 2026"},
    {"type": "insurance", "query": "insurtech insolvency failure 2026"},
    # Domain 4: Healthcare Facilities & Demographics
    {"type": "healthcare_re", "query": "hospital system closure rural facility CMS certification 2026"},
    {"type": "healthcare_re", "query": "outpatient migration surgery center development 2026"},
    {"type": "healthcare_re", "query": "physician practice acquisition private equity platform 2026"},
    {"type": "healthcare_re", "query": "Medicare reimbursement rate change facility payment 2026"},
    {"type": "healthcare_re", "query": "senior living development pipeline aging demographics 2026"},
    {"type": "healthcare_re", "query": "certificate of need CON application hospital expansion 2026"},
    {"type": "healthcare_re", "query": "nursing home closure state survey deficiency 2026"},
    # Domain 5: Energy & Data Centre Infrastructure
    {"type": "energy_infra", "query": "data center power demand utility capacity constraint 2026"},
    {"type": "energy_infra", "query": "hyperscaler lease agreement colocation expansion 2026"},
    {"type": "energy_infra", "query": "FERC interconnection queue backlog generation project 2026"},
    {"type": "energy_infra", "query": "utility capex rate case infrastructure investment 2026"},
    {"type": "energy_infra", "query": "grid reliability constraint transmission bottleneck 2026"},
    {"type": "energy_infra", "query": "natural gas power plant retirement replacement capacity 2026"},
    {"type": "energy_infra", "query": "behind the meter battery storage commercial installation 2026"},
    # Domain 6: Distressed Credit & Special Situations Catalysts
    {"type": "distressed", "query": "WARN Act notice large employer layoff plant closure 2026"},
    {"type": "distressed", "query": "PACER bankruptcy filing significant assets 2026"},
    {"type": "distressed", "query": "credit rating downgrade Moody S&P corporate 2026"},
    {"type": "distressed", "query": "UCC filing secured creditor lien perfection 2026"},
    {"type": "distressed", "query": "SEC restatement notification filing amendment 2026"},
    {"type": "distressed", "query": "activist investor 13D filing board replacement 2026"},
    {"type": "distressed", "query": "corporate spin-off divestiture announcement 2026"},
    {"type": "distressed", "query": "state tax incentive clawback default relocation 2026"},
    # === MODEL-TARGETING QUERIES ===
    # These pull facts that contain named models, specific assumptions, and calibration data.
    # The implication layer turns these into broken-model bridge vocabulary that competes
    # with insurance actuarial vocabulary in the collision evaluator.

    # === MULTI-LINGUAL SILO MINING ===
    # Language is the deepest professional silo: zero US analysts read these
    # sources in the original language, so compositions bridging US + (DE/JP/CN/KR)
    # are genuinely uncopyable without our stack.
    # Extraction prompt handles translation inline; facts stored in English
    # with original_language captured in raw_content prefix.

    # --- GERMAN (BaFin, Bundesgerichtshof, Bundestag, Deutsche Börse) ---
    {"type": "regulation", "query": "BaFin Verbraucherschutz Mitteilung Rundschreiben 2026"},
    {"type": "regulation", "query": "BaFin Aufsichtsverfahren Versicherung Solvency Richtlinie 2026"},
    {"type": "sec_filing", "query": "Bundesanzeiger Jahresabschluss DAX MDAX Ad-hoc-Meldung 2026"},
    {"type": "bankruptcy", "query": "Insolvenzbekanntmachungen Insolvenzverfahren Mittelstand 2026"},
    {"type": "patent", "query": "Deutsches Patent- und Markenamt erteilte Patente 2026 Chemie Pharma"},
    {"type": "distressed", "query": "BGH Urteil Schuldverschreibung Gläubigerrechte Restrukturierung 2026"},
    {"type": "earnings", "query": "Deutsche Bundesbank Wirtschaftsbericht Industriekonjunktur 2026"},
    {"type": "regulation", "query": "Bundesnetzagentur Festlegung Netzentgelte Stromnetz 2026"},

    # --- JAPANESE (金融庁, 東京証券取引所, EDINET, 特許庁) ---
    {"type": "sec_filing", "query": "EDINET 有価証券報告書 東証プライム 2026年 業績"},
    {"type": "regulation", "query": "金融庁 監督指針 改正 保険会社 2026年"},
    {"type": "regulation", "query": "厚生労働省 薬価改定 後発医薬品 2026年"},
    {"type": "patent", "query": "特許庁 特許公報 半導体 新規性 2026年"},
    {"type": "bankruptcy", "query": "東京地方裁判所 民事再生 会社更生 中堅企業 2026年"},
    {"type": "commodity", "query": "日本取引所 TOCOM 金 白金 相場 2026年"},
    {"type": "earnings", "query": "東証 決算短信 通期業績予想 修正 2026年"},
    {"type": "other", "query": "経済産業省 産業政策 重要技術 指定 2026年"},

    # --- CHINESE (证监会, 中国人民银行, 国家药监局, 国家知识产权局) ---
    {"type": "regulation", "query": "中国证监会 监管指引 上市公司 2026 修订"},
    {"type": "regulation", "query": "中国人民银行 货币政策 执行报告 2026 宏观审慎"},
    {"type": "regulation", "query": "国家药品监督管理局 药品审评 批准 2026 创新药"},
    {"type": "patent", "query": "国家知识产权局 发明专利 授权公告 2026 新能源"},
    {"type": "sec_filing", "query": "上交所 科创板 重大事项 信息披露 2026"},
    {"type": "bankruptcy", "query": "最高人民法院 破产重整 上市公司 2026"},
    {"type": "other", "query": "国家发展改革委 产业结构调整 目录 2026"},
    {"type": "commodity", "query": "上海期货交易所 金属 持仓 2026"},

    # --- KOREAN (금융감독원, DART, 특허청, 식약처) ---
    {"type": "sec_filing", "query": "DART 공시 유가증권보고서 코스피 2026"},
    {"type": "regulation", "query": "금융감독원 검사 제재 보험업 2026"},
    {"type": "regulation", "query": "식품의약품안전처 의약품 허가 2026 신약"},
    {"type": "patent", "query": "특허청 특허 등록 반도체 이차전지 2026"},
    {"type": "bankruptcy", "query": "서울회생법원 회생절차 상장사 2026"},
    {"type": "other", "query": "산업통상자원부 소재부품장비 정책 2026"},

    # --- FRENCH (AMF, Banque de France, ACPR) — bonus ---
    {"type": "regulation", "query": "AMF Autorité marchés financiers position recommandation 2026"},
    {"type": "regulation", "query": "ACPR Autorité prudentielle contrôle résolution 2026 Solvabilité"},
    {"type": "bankruptcy", "query": "BODACC annonces redressement liquidation ETI 2026"},

    # CRE VALUATION METHODOLOGY (10 queries)
    {"type": "cre_credit", "query": "ARGUS DCF model assumptions cap rate spread office REIT valuation methodology 2025 2026"},
    {"type": "cre_credit", "query": "CMBS underwriting DSCR debt service coverage ratio assumptions interest rate cap expiry 2025 2026"},
    {"type": "cre_credit", "query": "Green Street CPPI commercial property price index methodology adjustment 2025 2026"},
    {"type": "cre_credit", "query": "appraisal cap rate selection methodology comparable sales adjustment commercial real estate 2025"},
    {"type": "cre_credit", "query": "tenant improvement allowance TI concession increase cost per square foot office lease 2024 2025"},
    {"type": "specialty_re", "query": "NIC senior housing valuation NOI methodology occupancy stabilization assumption 2025 2026"},
    {"type": "specialty_re", "query": "data center valuation methodology power density pricing per megawatt lease structure 2025 2026"},
    {"type": "specialty_re", "query": "industrial warehouse last mile logistics rent growth assumption CBRE cap rate survey 2025"},
    {"type": "cre_credit", "query": "real estate debt fund IRR waterfall model assumption default rate recovery rate 2025"},
    {"type": "cre_credit", "query": "office building residual value assumption conversion cost estimate adaptive reuse feasibility 2025 2026"},

    # CONSTRUCTION COST ESTIMATION METHODOLOGY (10 queries)
    {"type": "commodity", "query": "RSMeans construction cost database accuracy federal infrastructure prevailing wage Davis Bacon 2025"},
    {"type": "commodity", "query": "ENR construction cost index methodology material labor split accuracy 2025 2026"},
    {"type": "commodity", "query": "construction bid estimate accuracy overrun underrun federal highway project FHWA 2025"},
    {"type": "commodity", "query": "modular construction cost comparison traditional stick built per square foot 2025"},
    {"type": "commodity", "query": "IRA prevailing wage apprenticeship requirement construction cost impact federal project 2025 2026"},
    {"type": "commodity", "query": "construction material lead time procurement delay steel concrete lumber delivery 2025"},
    {"type": "commodity", "query": "contractor surety bond capacity utilization default rate construction 2025"},
    {"type": "commodity", "query": "construction labor productivity decline rate Bureau of Labor Statistics 2024 2025"},
    {"type": "commodity", "query": "building permit cost escalation municipal fee impact development pro forma 2025"},
    {"type": "commodity", "query": "mass timber CLT cost comparison concrete steel structural frame commercial building 2025"},

    # SHIPPING & FREIGHT PRICING METHODOLOGY (10 queries)
    {"type": "commodity", "query": "Baltic Dry Index BDI forward freight agreement FFA pricing methodology vessel supply model 2025 2026"},
    {"type": "commodity", "query": "container shipping rate SCFI methodology Drewry forecast accuracy 2025"},
    {"type": "commodity", "query": "EEXI CII compliance vessel scrapping rate fleet supply model assumption 2025 2026"},
    {"type": "commodity", "query": "bunker fuel cost hedging methodology VLSFO MGO spread assumption shipowner 2025"},
    {"type": "commodity", "query": "port congestion dwell time cost model assumption container terminal throughput 2025"},
    {"type": "commodity", "query": "Panama Canal draft restriction transit slot auction pricing model 2025 2026"},
    {"type": "commodity", "query": "dry bulk Capesize time charter rate earnings model iron ore coal trade flow 2025"},
    {"type": "commodity", "query": "tanker fleet age profile scrapping forecast orderbook delivery schedule Clarksons 2025"},
    {"type": "commodity", "query": "intermodal rail truck cost comparison per ton mile routing model assumption 2025"},
    {"type": "commodity", "query": "warehouse logistics last mile delivery cost model per package urban suburban rural 2025"},

    # BANKING CREDIT UNDERWRITING METHODOLOGY (10 queries)
    {"type": "cre_credit", "query": "bank CRE loan underwriting DSCR debt yield stressed interest rate assumption 2025 2026"},
    {"type": "cre_credit", "query": "CECL current expected credit loss model assumption CRE portfolio vintage analysis 2025"},
    {"type": "cre_credit", "query": "OCC Comptroller guidance CRE concentration risk stress test methodology 2025 2026"},
    {"type": "distressed", "query": "leveraged loan default rate recovery rate assumption CLO model Moody S&P 2025"},
    {"type": "distressed", "query": "middle market direct lending underwriting standard covenant lite assumption 2025"},
    {"type": "distressed", "query": "credit rating model transition matrix migration probability methodology update 2025"},
    {"type": "distressed", "query": "trade credit insurance pricing model supplier default probability assumption Euler Hermes Coface 2025"},
    {"type": "distressed", "query": "distressed debt recovery rate waterfall model secured unsecured priority 2025"},
    {"type": "earnings", "query": "bank net interest margin NIM model assumption deposit beta rate sensitivity 2025 2026"},
    {"type": "earnings", "query": "community bank CRE concentration ratio FDIC supervisory threshold 300 percent 2025"},
]

# Source type display icons
SOURCE_ICONS = {
    "patent": "📜",
    "sec_filing": "📊",
    "government_contract": "🏛️",
    "regulation": "⚖️",
    "bankruptcy": "💀",
    "commodity": "📈",
    "pharmaceutical": "💊",
    "academic": "🔬",
    "job_listing": "👥",
    "app_ranking": "📱",
    "earnings": "💰",
    "other": "📌",
    "cre_credit": "🏢",
    "specialty_re": "🏗️",
    "insurance": "🛡️",
    "healthcare_re": "🏥",
    "energy_infra": "⚡",
    "distressed": "🔻",
}

# === Domain Distance Matrix (Complete 18×18 = 153 pairs) ===
# Each cell: 0.0 (same silo) to 1.0 (maximally distant professional worlds)
# Measures information-theoretic independence, not topical similarity.
# Question: "Would the same analyst plausibly read both source types?"
#   0.1-0.2 = same desk (analyst reads both daily)
#   0.3-0.4 = same floor (occasional cross-read)
#   0.5-0.6 = same building (aware each exists, rarely reads)
#   0.7-0.8 = different city (would never encounter)
#   0.9-1.0 = different planet (zero shared readership)
DOMAIN_DISTANCE = {
    # ── WALL STREET CLUSTER ──
    # Equity analysts, credit analysts, distressed debt traders
    ("sec_filing", "earnings"):           0.15,  # same analyst, different tabs
    ("sec_filing", "distressed"):         0.25,  # credit desk reads both
    ("sec_filing", "cre_credit"):         0.35,  # REIT analysts cross over
    ("sec_filing", "specialty_re"):       0.45,  # REIT analyst knows specialty
    ("sec_filing", "healthcare_re"):      0.50,  # healthcare REIT is niche
    ("sec_filing", "energy_infra"):       0.45,  # utility/infra analysts read filings
    ("sec_filing", "insurance"):          0.55,  # insurance equity analysts exist but niche
    ("sec_filing", "regulation"):         0.40,  # compliance reads both
    ("sec_filing", "bankruptcy"):         0.40,  # restructuring teams read filings
    ("sec_filing", "commodity"):          0.50,  # commodity equities bridge
    ("sec_filing", "pharmaceutical"):     0.55,  # pharma equity analysts
    ("sec_filing", "patent"):             0.75,  # patent attorneys don't read 10-Ks
    ("sec_filing", "academic"):           0.75,  # academics rarely read EDGAR
    ("sec_filing", "government_contract"):0.60,  # defense analysts bridge
    ("sec_filing", "job_listing"):        0.50,  # HR signal for equity analysts
    ("sec_filing", "app_ranking"):        0.60,  # tech analysts might check
    ("sec_filing", "other"):              0.55,  # lawsuits, environmental — some crossover

    ("earnings", "distressed"):           0.30,  # credit analysts track earnings
    ("earnings", "cre_credit"):           0.35,  # REIT earnings = CRE data
    ("earnings", "specialty_re"):         0.45,  # specialty REIT earnings
    ("earnings", "healthcare_re"):        0.50,  # healthcare REIT earnings
    ("earnings", "energy_infra"):         0.45,  # utility earnings
    ("earnings", "insurance"):            0.55,  # insurance earnings niche
    ("earnings", "regulation"):           0.50,  # regulatory impact on earnings
    ("earnings", "bankruptcy"):           0.45,  # earnings miss → distress
    ("earnings", "commodity"):            0.50,  # commodity company earnings
    ("earnings", "pharmaceutical"):       0.55,  # pharma earnings
    ("earnings", "patent"):               0.80,  # patent office ≠ earnings calls
    ("earnings", "academic"):             0.80,  # academics don't listen to calls
    ("earnings", "government_contract"):  0.65,  # defense contractor earnings
    ("earnings", "job_listing"):          0.45,  # hiring signals for equities
    ("earnings", "app_ranking"):          0.55,  # tech earnings + app data
    ("earnings", "other"):                0.55,  # general news affects earnings

    ("distressed", "cre_credit"):         0.30,  # CRE distress = core overlap
    ("distressed", "specialty_re"):       0.40,  # distressed specialty assets
    ("distressed", "healthcare_re"):      0.45,  # distressed healthcare facilities
    ("distressed", "energy_infra"):       0.50,  # distressed energy projects
    ("distressed", "insurance"):          0.50,  # insurance on distressed credits
    ("distressed", "regulation"):         0.50,  # regulatory triggers for distress
    ("distressed", "bankruptcy"):         0.25,  # same lawyers, same courtrooms
    ("distressed", "commodity"):          0.60,  # commodity company distress
    ("distressed", "pharmaceutical"):     0.70,  # pharma distress is niche
    ("distressed", "patent"):             0.85,  # patent office ≠ bankruptcy court
    ("distressed", "academic"):           0.80,  # academics don't track Chapter 11
    ("distressed", "government_contract"):0.70,  # government contractor distress rare
    ("distressed", "job_listing"):        0.50,  # WARN Act = distress signal
    ("distressed", "app_ranking"):        0.75,  # app rankings ≠ restructuring
    ("distressed", "other"):              0.55,  # lawsuits, environmental claims

    # ── REAL ESTATE CLUSTER ──
    # CRE analysts, appraisers, property managers, developers
    ("cre_credit", "specialty_re"):       0.20,  # same CRE desk, different asset class
    ("cre_credit", "healthcare_re"):      0.30,  # healthcare is CRE sub-specialty
    ("cre_credit", "energy_infra"):       0.50,  # data center bridges both
    ("cre_credit", "insurance"):          0.50,  # insurance CRE allocation
    ("cre_credit", "regulation"):         0.45,  # zoning, building codes
    ("cre_credit", "bankruptcy"):         0.45,  # CRE bankruptcy proceedings
    ("cre_credit", "commodity"):          0.55,  # construction materials bridge
    ("cre_credit", "pharmaceutical"):     0.90,  # zero overlap
    ("cre_credit", "patent"):             0.85,  # zero overlap
    ("cre_credit", "academic"):           0.80,  # urban planning academics, rare
    ("cre_credit", "government_contract"):0.65,  # government facilities
    ("cre_credit", "job_listing"):        0.55,  # REIT hiring signals
    ("cre_credit", "app_ranking"):        0.85,  # zero overlap
    ("cre_credit", "other"):              0.55,  # environmental, lawsuits on property

    ("specialty_re", "healthcare_re"):    0.35,  # both specialty property types
    ("specialty_re", "energy_infra"):     0.40,  # data centers bridge both
    ("specialty_re", "insurance"):        0.60,  # insurance on specialty assets
    ("specialty_re", "regulation"):       0.50,  # zoning, environmental
    ("specialty_re", "bankruptcy"):       0.55,  # distressed specialty assets
    ("specialty_re", "commodity"):        0.60,  # construction input costs
    ("specialty_re", "pharmaceutical"):   0.85,  # zero overlap
    ("specialty_re", "patent"):           0.90,  # zero overlap
    ("specialty_re", "academic"):         0.80,  # rare academic crossover
    ("specialty_re", "government_contract"):0.60, # government-leased facilities
    ("specialty_re", "job_listing"):      0.55,  # property manager hiring
    ("specialty_re", "app_ranking"):      0.80,  # zero overlap
    ("specialty_re", "other"):            0.60,  # environmental, infrastructure

    ("healthcare_re", "energy_infra"):    0.65,  # different infrastructure types
    ("healthcare_re", "insurance"):       0.55,  # health insurance ≠ facility RE
    ("healthcare_re", "regulation"):      0.40,  # CMS, CON laws heavily regulate
    ("healthcare_re", "bankruptcy"):      0.65,  # hospital closures
    ("healthcare_re", "commodity"):       0.75,  # construction costs only bridge
    ("healthcare_re", "pharmaceutical"):  0.45,  # pharma + healthcare facility overlap
    ("healthcare_re", "patent"):          0.80,  # medical device patents, rare
    ("healthcare_re", "academic"):        0.60,  # medical school research
    ("healthcare_re", "government_contract"):0.55, # VA hospitals, Medicare
    ("healthcare_re", "job_listing"):     0.55,  # healthcare staffing signals
    ("healthcare_re", "app_ranking"):     0.80,  # zero overlap
    ("healthcare_re", "other"):           0.60,  # healthcare lawsuits, environmental

    # ── ENERGY / INFRASTRUCTURE ──
    ("energy_infra", "insurance"):        0.60,  # utility insurance, cat bonds
    ("energy_infra", "regulation"):       0.40,  # FERC, state PUCs heavily regulate
    ("energy_infra", "bankruptcy"):       0.65,  # energy company bankruptcies
    ("energy_infra", "commodity"):        0.35,  # gas prices = power prices
    ("energy_infra", "pharmaceutical"):   0.90,  # zero overlap
    ("energy_infra", "patent"):           0.75,  # renewable energy patents
    ("energy_infra", "academic"):         0.65,  # climate/grid research
    ("energy_infra", "government_contract"):0.50, # DOE, grid infrastructure
    ("energy_infra", "job_listing"):      0.55,  # lineman shortage signals
    ("energy_infra", "app_ranking"):      0.75,  # zero overlap
    ("energy_infra", "other"):            0.55,  # environmental, infrastructure failures

    # ── INSURANCE ──
    # Actuaries, underwriters, state commissioners — their own professional world
    ("insurance", "regulation"):          0.45,  # state insurance regulators
    ("insurance", "bankruptcy"):          0.60,  # surety, insolvency
    ("insurance", "commodity"):           0.70,  # cat bonds ↔ weather, thin bridge
    ("insurance", "pharmaceutical"):      0.65,  # health/life insurance + pharma
    ("insurance", "patent"):              0.85,  # zero overlap
    ("insurance", "academic"):            0.75,  # actuarial science
    ("insurance", "government_contract"): 0.70,  # government insurance programs
    ("insurance", "job_listing"):         0.65,  # actuary hiring
    ("insurance", "app_ranking"):         0.80,  # insurtech apps, thin bridge
    ("insurance", "other"):               0.65,  # lawsuits, environmental claims

    # ── REGULATION ──
    # Compliance officers, regulatory attorneys, policy analysts
    ("regulation", "bankruptcy"):         0.55,  # regulatory triggers for distress
    ("regulation", "commodity"):          0.50,  # EPA, trade rules affect commodities
    ("regulation", "pharmaceutical"):     0.40,  # FDA regulations
    ("regulation", "patent"):             0.60,  # IP law ≠ regulatory law
    ("regulation", "academic"):           0.55,  # policy research
    ("regulation", "government_contract"):0.35,  # procurement rules = regulation
    ("regulation", "job_listing"):        0.60,  # compliance hiring
    ("regulation", "app_ranking"):        0.75,  # platform regulation, thin bridge
    ("regulation", "other"):              0.50,  # environmental law, zoning

    # ── BANKRUPTCY / RESTRUCTURING ──
    ("bankruptcy", "commodity"):          0.70,  # commodity company bankruptcy
    ("bankruptcy", "pharmaceutical"):     0.75,  # pharma bankruptcy rare
    ("bankruptcy", "patent"):             0.80,  # IP auctions in bankruptcy
    ("bankruptcy", "academic"):           0.80,  # academics don't read PACER
    ("bankruptcy", "government_contract"):0.70,  # government contractor distress
    ("bankruptcy", "job_listing"):        0.50,  # WARN Act filings
    ("bankruptcy", "app_ranking"):        0.70,  # tech startup bankruptcy
    ("bankruptcy", "other"):              0.55,  # lawsuits trigger bankruptcy

    # ── COMMODITY ──
    # Traders, supply chain managers, mining engineers
    ("commodity", "pharmaceutical"):      0.80,  # API raw materials, thin bridge
    ("commodity", "patent"):              0.70,  # materials patents
    ("commodity", "academic"):            0.65,  # materials science
    ("commodity", "government_contract"): 0.55,  # government stockpiles
    ("commodity", "job_listing"):         0.70,  # mining/logistics hiring
    ("commodity", "app_ranking"):         0.85,  # zero overlap
    ("commodity", "other"):               0.60,  # supply chain disruptions

    # ── PHARMACEUTICAL ──
    # FDA regulatory affairs, clinical researchers, pharma patent attorneys
    ("pharmaceutical", "patent"):         0.50,  # pharma patent attorneys exist
    ("pharmaceutical", "academic"):       0.35,  # clinical research ↔ academic
    ("pharmaceutical", "government_contract"):0.70, # NIH grants, rare
    ("pharmaceutical", "job_listing"):    0.70,  # regulatory specialist hiring
    ("pharmaceutical", "app_ranking"):    0.85,  # zero overlap
    ("pharmaceutical", "other"):          0.70,  # pharma lawsuits

    # ── PATENT OFFICE ──
    # IP attorneys, patent examiners, tech transfer offices
    ("patent", "academic"):               0.45,  # university tech transfer
    ("patent", "government_contract"):    0.65,  # SBIR/STTR patents
    ("patent", "job_listing"):            0.85,  # zero overlap
    ("patent", "app_ranking"):            0.90,  # zero overlap
    ("patent", "other"):                  0.75,  # IP litigation

    # ── ACADEMIC ──
    # Researchers, professors, peer reviewers
    ("academic", "government_contract"):  0.50,  # NSF/DOE funded research
    ("academic", "job_listing"):          0.75,  # faculty hiring, different world
    ("academic", "app_ranking"):          0.80,  # zero overlap
    ("academic", "other"):                0.65,  # retractions, environmental studies

    # ── GOVERNMENT CONTRACT ──
    ("government_contract", "job_listing"):0.55,  # federal hiring signals
    ("government_contract", "app_ranking"):0.75,  # govtech apps, thin bridge
    ("government_contract", "other"):     0.60,  # government lawsuits, environmental

    # ── JOB LISTING ──
    # HR analytics, labor market researchers
    ("job_listing", "app_ranking"):       0.60,  # tech hiring + app growth
    ("job_listing", "other"):             0.55,  # layoff news, class actions

    # ── APP RANKING ──
    ("app_ranking", "other"):             0.65,  # app lawsuits, thin bridge
}


def get_domain_distance(source_a, source_b):
    """Get the information-theoretic distance between two source types.
    Returns 0.0 for same source, looks up matrix for known pairs,
    defaults to 0.6 for unknown pairs."""
    if source_a == source_b:
        return 0.0
    # Check both orderings
    key = (source_a, source_b)
    if key in DOMAIN_DISTANCE:
        return DOMAIN_DISTANCE[key]
    key_rev = (source_b, source_a)
    if key_rev in DOMAIN_DISTANCE:
        return DOMAIN_DISTANCE[key_rev]
    # Default: assume moderate distance for unmapped pairs
    return 0.6


def compute_avg_domain_distance(source_types):
    """Compute average pairwise distance for a set of source types.
    Returns a value from 0.0 (all same silo) to 1.0 (maximally distant).
    Used to replace flat domain counting with actual silo distance."""
    types = list(source_types)
    if len(types) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            total += get_domain_distance(types[i], types[j])
            pairs += 1
    return total / pairs if pairs > 0 else 0.0


# === Temporal Asymmetry Scoring ===
TEMPORAL_SPREAD_THRESHOLDS = {"high": 180, "medium": 90, "low": 30}
TEMPORAL_AGE_THRESHOLDS = {"high": 270, "medium": 180, "low": 90}
TEMPORAL_BONUS_CAP = 8

# === Negative Space Detection ===
NEGATIVE_SPACE_ENABLED = True
NEGATIVE_SPACE_BONUS_CAP = 7
NEGATIVE_SPACE_PENALTY_MAX = -2

# === Information Decay Model (Branch 4) ===
# Channel prestige tiers: maps discovery channel keywords to daily decay rate
# Rates derived from half-life model: rate = 1 - 0.5^(1/half_life_days)
# Decay formula: edge_remaining = (1 - decay_rate) ^ days_since_discovery
# Penalty = -round(15 * (1 - edge_remaining))
#
# Half-life examples:
#   Goldman note: ~0.5 day half-life → 3 days later penalty = -15 (instant death)
#   Seeking Alpha: ~3 day half-life → 3 days later penalty = -7
#   Newsletter: ~14 day half-life → 30 days later penalty = -12
#   Blog/tweet: ~60 day half-life → 30 days later penalty = -4
import math as _math
def _hl_rate(half_life_days):
    return 1.0 - 0.5 ** (1.0 / half_life_days)

CHANNEL_DECAY_TIERS = {
    # Tier 1: Institutional terminal-speed (half-life ~0.5 days)
    "bloomberg": _hl_rate(0.5), "reuters": _hl_rate(0.5), "goldman sachs": _hl_rate(0.5),
    "jpmorgan": _hl_rate(0.5), "morgan stanley": _hl_rate(0.5), "financial times": _hl_rate(0.5),
    "wall street journal": _hl_rate(0.5), "wsj": _hl_rate(0.5), "ft.com": _hl_rate(0.5),
    # Tier 2: Professional sell-side / high-prestige editorial (half-life ~3 days)
    "seeking alpha": _hl_rate(3), "barrons": _hl_rate(3), "barron's": _hl_rate(3),
    "morningstar": _hl_rate(3), "citi research": _hl_rate(3), "ubs": _hl_rate(3),
    "deutsche bank": _hl_rate(3), "bmo": _hl_rate(3), "rbc": _hl_rate(3),
    "jefferies": _hl_rate(3), "cowen": _hl_rate(3), "fierce pharma": _hl_rate(3),
    "biopharma dive": _hl_rate(3), "cnbc": _hl_rate(3),
    # Tier 3: Industry-specific / specialist (half-life ~14 days)
    "trade publication": _hl_rate(14), "industry newsletter": _hl_rate(14),
    "specialist blog": _hl_rate(14), "substack": _hl_rate(14), "industry report": _hl_rate(14),
    "sector report": _hl_rate(14), "conference presentation": _hl_rate(14),
    # Tier 4: Fringe / low-reach (half-life ~60 days)
    "tweet": _hl_rate(60), "twitter": _hl_rate(60), "x.com": _hl_rate(60), "reddit": _hl_rate(60),
    "forum": _hl_rate(60), "comment": _hl_rate(60), "podcast": _hl_rate(60), "blog post": _hl_rate(60),
}
DECAY_DEFAULT_RATE = _hl_rate(7)  # Unknown channel: ~7 day half-life
DECAY_MAX_PENALTY = 15


def compute_edge_decay_penalty(discovery_channels, time_since_days):
    """Compute continuous decay penalty based on highest-prestige channel and time.
    Returns negative integer penalty (0 to -DECAY_MAX_PENALTY).
    Goldman note 3 days ago = -14. Blog 30 days ago = -5. No coverage = 0."""
    if not discovery_channels or time_since_days is None:
        return 0
    max_decay_rate = 0.0
    for channel in discovery_channels:
        channel_lower = channel.lower().strip()
        for keyword, rate in CHANNEL_DECAY_TIERS.items():
            if keyword in channel_lower:
                max_decay_rate = max(max_decay_rate, rate)
                break
    if max_decay_rate == 0.0:
        max_decay_rate = DECAY_DEFAULT_RATE
    days = max(0, float(time_since_days))
    edge_remaining = (1.0 - max_decay_rate) ** days
    penalty = -round(DECAY_MAX_PENALTY * (1.0 - edge_remaining))
    return max(-DECAY_MAX_PENALTY, min(0, penalty))


# === Adversarial Corpus Injection (Branch 5) ===
GAP_TARGETING_RATIO = 0.20  # 20% of ingest cycles use gap-targeted queries

# === Self-Balancing System ===

# Focus mode: "default" = maximum breadth, "core" = anchor in priority source verticals
FOCUS_MODE = "core"

# Core / priority source types (canonical definition — dashboard imports from here)
CORE_SOURCE_TYPES = {
    "cre_credit", "specialty_re", "insurance",
    "healthcare_re", "energy_infra", "distressed",
}

# Rotation: check last N survived hypotheses for domain-pair repeats
ROTATION_WINDOW = 5
# Graduated penalty: 0.7 for first repeat, 0.4 for second+
ROTATION_PENALTY_FIRST = 0.7
ROTATION_PENALTY_SECOND = 0.4

# Feedback loop: blend collision productivity into source selection weights
# 0.4 = 40% productivity signal, 60% existing deficit signal
PRODUCTIVITY_BLEND_WEIGHT = 0.4
# Minimum facts before a source type is included in productivity scoring
PRODUCTIVITY_MIN_FACTS = 30

# === Theory Run Configuration ===
THEORY_RUN = False
THEORY_RUN_CYCLES = 10
THEORY_RUN_COST_LIMIT_USD = 50.0
THEORY_RUN_BATCH_ID = "theory_run_01"
THEORY_RUN_DB = "hunter_theory_run_01.db"
THEORY_RUN_SELF_DOMAINS = []  # User's publishing domains for self-contribution probe

# === v3 Golden — Retrospective Validation Experiment ===
# Retrospective mode
RETROSPECTIVE_MODE = False
RETROSPECTIVE_CUTOFF_DATE = "2024-12-31"
RETROSPECTIVE_DISABLE_WEB_SEARCH = True  # disable web search in kill gauntlet
RETROSPECTIVE_TEMPERATURE = 0.3
RETROSPECTIVE_ENFORCE_FACT_CITATION = True  # compile-time leakage prevention
RETROSPECTIVE_DATE_FIELD = "date_of_fact"  # determined by diagnostic
RETROSPECTIVE_EXCLUDE_UNKNOWN_DATES = True  # exclude facts with "unknown" dates

# Module I: Single-domain control batch flag
FORCE_SINGLE_DOMAIN = False
COLLISION_MIN_DOMAINS = 2   # minimum distinct source types for collision
COLLISION_MAX_DOMAINS = None  # None = unbounded

# Stratum definitions
STRATA_CONFIG = {
    "A": {"force_single_domain": True, "min_domains": 1, "max_domains": 1, "label": "single_domain"},
    "B": {"force_single_domain": False, "min_domains": 2, "max_domains": 2, "label": "cross_2"},
    "C": {"force_single_domain": False, "min_domains": 3, "max_domains": 3, "label": "cross_3"},
    "D": {"force_single_domain": False, "min_domains": 2, "max_domains": None, "label": "cross_open"},
}

# Current stratum (set dynamically by execution harness)
CURRENT_STRATUM = None
CURRENT_COLLISION_MODE = None

# v3 Golden budget and execution
V3_GOLDEN_COST_LIMIT_USD = 600.0
V3_GOLDEN_CYCLES = 240  # 60 per stratum
V3_GOLDEN_TIME_LIMIT_HOURS = 14
V3_GOLDEN_DB = "hunter_v3_golden.db"
V3_GOLDEN_BATCH_ID = "v3_golden"
V3_GOLDEN_OUTPUT_DIR = "theory_run_v3_golden_output"
