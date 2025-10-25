"""
Professional Trading Dashboard - Advanced Modern Style
Inspired by Sky Central's clean, professional design with Urbanist font
"""

def get_modern_css() -> str:
    """Returns advanced CSS with professional blue theme and glassmorphism"""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Urbanist:wght@300;400;500;600;700;800;900&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        :root {
            /* Professional Blue Palette (from Sky Central) */
            --primary: hsl(217, 91%, 60%);
            --primary-foreground: #FFFFFF;
            --primary-glow: hsl(217, 91%, 66%);
            --primary-deep: hsl(220, 26%, 22%);
            
            /* Clean Backgrounds */
            --bg-primary: hsl(215, 25%, 96%);
            --bg-secondary: #FFFFFF;
            --bg-tertiary: hsl(215, 25%, 94%);
            --bg-card: #FFFFFF;
            
            /* Professional Colors */
            --secondary: #4C9AFF;
            --accent: #4C9AFF;
            --success: hsl(160, 84%, 39%);
            --warning: hsl(38, 92%, 50%);
            --danger: #FF5630;
            --info: #4C9AFF;
            
            /* Text Colors */
            --text-primary: hsl(220, 26%, 14%);
            --text-secondary: hsl(215, 20%, 40%);
            --text-tertiary: hsl(215, 20%, 65%);
            --text-white: #FFFFFF;
            
            /* Gradients */
            --gradient-primary: linear-gradient(135deg, #0046A3 0%, #007BFF 100%);
            --gradient-secondary: linear-gradient(135deg, #0052CC 0%, #4C9AFF 100%);
            --gradient-hero: linear-gradient(135deg, #0046A3 0%, #0052CC 50%, #4C9AFF 100%);
            
            /* Shadows - Professional depth */
            --shadow-sm: 0 1px 2px 0 rgba(0, 82, 204, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 82, 204, 0.1), 0 2px 4px -1px rgba(0, 82, 204, 0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0, 82, 204, 0.1), 0 4px 6px -2px rgba(0, 82, 204, 0.05);
            --shadow-xl: 0 20px 25px -5px rgba(0, 82, 204, 0.1), 0 10px 10px -5px rgba(0, 82, 204, 0.04);
            --shadow-glow: 0 0 20px rgba(0, 82, 204, 0.3);
            --shadow-card: 0 8px 24px rgba(0, 82, 204, 0.12);
        }
        
        /* ============================================
           GLOBAL FONT & STYLING
           ============================================ */
        * {
            font-family: 'Urbanist', 'Century Gothic', 'Futura', 'Avant Garde', -apple-system, sans-serif !important;
        }
        
        .stApp {
            background: linear-gradient(180deg,
                hsl(215, 25%, 98%) 0%,
                hsl(215, 25%, 96%) 50%,
                hsl(215, 25%, 94%) 100%) !important;
            background-attachment: fixed !important;
        }
        
        /* Subtle orbs for depth */
        .stApp::before {
            content: '';
            position: fixed;
            top: -20%;
            left: -20%;
            width: 80vw;
            height: 80vw;
            background: radial-gradient(circle at 50% 50%, rgba(107, 182, 255, 0.12), transparent 70%);
            filter: blur(80px);
            opacity: 0.4;
            animation: pulseGlow 8s ease-in-out infinite;
            pointer-events: none;
            z-index: 0;
        }
        
        @keyframes pulseGlow {
            0%, 100% { transform: scale(1); opacity: 0.4; }
            50% { transform: scale(1.08); opacity: 0.6; }
        }
        
        .main {
            background: transparent;
            padding: 2rem;
            font-family: 'Urbanist', sans-serif;
            color: var(--text-primary);
            position: relative;
            z-index: 1;
        }
        
        /* ============================================
           METRIC CARDS - GLASSMORPHISM STYLE
           ============================================ */
        [data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(20px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
            border-radius: 1.5rem !important;
            padding: 1.5rem !important;
            box-shadow: 
                0 8px 32px rgba(0, 82, 204, 0.12),
                0 4px 16px rgba(96, 165, 250, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.6) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        [data-testid="metric-container"]::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.8) 50%, transparent 100%);
            opacity: 0.6;
        }
        
        [data-testid="metric-container"]:hover {
            transform: translateY(-3px) scale(1.005) !important;
            box-shadow: 
                0 24px 80px rgba(0, 82, 204, 0.15),
                0 12px 32px rgba(96, 165, 250, 0.1),
                inset 0 1px 2px rgba(255, 255, 255, 1) !important;
        }
        
        [data-testid="metric-container"] [data-testid="metric-label"] {
            color: var(--text-secondary) !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }
        
        [data-testid="metric-container"] [data-testid="metric-value"] {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            font-size: 2rem !important;
        }
        
        [data-testid="metric-container"] [data-testid="metric-delta"] {
            font-weight: 600 !important;
        }
        
        /* Special metric card styles */
        .metric-card-blue {
            background: linear-gradient(135deg, #EBF5FF 0%, #DBEAFE 100%) !important;
            border-left: 4px solid #5B8DEF !important;
        }
        
        .metric-card-green {
            background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%) !important;
            border-left: 4px solid #10B981 !important;
        }
        
        .metric-card-orange {
            background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%) !important;
            border-left: 4px solid #F59E0B !important;
        }
        
        .metric-card-purple {
            background: linear-gradient(135deg, #FAF5FF 0%, #EDE9FE 100%) !important;
            border-left: 4px solid #8B5CF6 !important;
        }
        
        /* ============================================
           HEADERS - CLEAN TYPOGRAPHY
           ============================================ */
        h1 {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            font-size: 2.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        h2 {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-size: 1.875rem !important;
            margin-bottom: 1rem !important;
        }
        
        h3 {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-size: 1.25rem !important;
            margin-bottom: 0.75rem !important;
        }
        
        /* ============================================
           BUTTONS - PROFESSIONAL BLUE THEME
           ============================================ */
        .stButton > button {
            background: #0052CC !important;
            color: white !important;
            border: none !important;
            border-radius: 0.5rem !important;
            padding: 0.625rem 1.25rem !important;
            font-weight: 600 !important;
            font-size: 0.875rem !important;
            box-shadow: 
                0 1px 3px rgba(0, 0, 0, 0.12),
                0 1px 2px rgba(0, 0, 0, 0.24) !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            letter-spacing: 0.025em !important;
        }
        
        .stButton > button:hover {
            background: #4C9AFF !important;
            transform: translateY(-1px) !important;
            box-shadow: 
                0 7px 14px rgba(76, 154, 255, 0.3),
                0 3px 6px rgba(0, 0, 0, 0.08) !important;
        }
        
        .stButton > button:active {
            background: #0046A3 !important;
            transform: translateY(0) !important;
            box-shadow: 
                0 1px 2px rgba(0, 0, 0, 0.12) !important;
        }
        
        /* Primary button style */
        [data-testid="stButton"][kind="primary"] > button,
        button[kind="primary"] {
            background: #ef4444 !important;
            box-shadow: 
                0 4px 6px rgba(239, 68, 68, 0.25),
                0 1px 3px rgba(0, 0, 0, 0.08) !important;
        }
        
        [data-testid="stButton"][kind="primary"] > button:hover,
        button[kind="primary"]:hover {
            background: #dc2626 !important;
        }
        
        /* ============================================
           TABS - PROFESSIONAL BLUE THEME
           ============================================ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: linear-gradient(135deg, rgba(0, 82, 204, 0.05), rgba(76, 154, 255, 0.02));
            border-bottom: 2px solid rgba(0, 82, 204, 0.2);
            border-radius: 12px 12px 0 0;
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: var(--text-secondary);
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-size: 0.875rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            color: #0052CC;
            background: linear-gradient(135deg, rgba(0, 82, 204, 0.1), rgba(76, 154, 255, 0.05));
            border-color: rgba(0, 82, 204, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 82, 204, 0.15);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #0052CC 0%, #0046A3 100%) !important;
            color: white !important;
            border: 1px solid #0046A3 !important;
            box-shadow: 0 4px 16px rgba(0, 82, 204, 0.3) !important;
            transform: translateY(-2px) !important;
        }
        
        .stTabs [aria-selected="true"]::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.5));
            animation: shimmer 2s infinite;
        }
        
        /* Enhanced subtabs for nested tabs */
        .stTabs .stTabs [data-baseweb="tab-list"] {
            background: rgba(0, 82, 204, 0.03);
            border-bottom: 1px solid rgba(0, 82, 204, 0.1);
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .stTabs .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem;
            padding: 0.5rem 1rem;
            margin: 0 0.25rem;
        }
        
        .stTabs .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(0, 82, 204, 0.8), rgba(0, 70, 163, 0.9)) !important;
        }
        
        /* Tab content styling */
        .stTabs > div > div[data-baseweb="tab-panel"] {
            background: rgba(0, 82, 204, 0.02);
            border-radius: 0 0 12px 12px;
            padding: 1.5rem;
            margin-top: -1px;
            border: 1px solid rgba(0, 82, 204, 0.1);
            border-top: none;
        }
        
        /* Nested tab content */
        .stTabs .stTabs > div > div[data-baseweb="tab-panel"] {
            background: rgba(0, 82, 204, 0.01);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            border: 1px solid rgba(0, 82, 204, 0.08);
        }
        
        /* ============================================
           CARDS & CONTAINERS - CLEAN STYLE
           ============================================ */
        .stContainer > div {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: var(--shadow-card);
            border: 1px solid rgba(226, 232, 240, 0.8);
        }
        
        /* Chart containers */
        .chart-container, .stPlotlyChart {
            background: var(--bg-card) !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            box-shadow: var(--shadow-card) !important;
            border: 1px solid rgba(139, 92, 246, 0.1) !important;
            margin-bottom: 1.5rem !important;
        }
        
        /* ============================================
           DATA TABLES - MODERN PURPLE THEME
           ============================================ */
        .stDataFrame {
            background: var(--bg-card) !important;
            border: none !important;
            border-radius: 16px !important;
            overflow: hidden !important;
            box-shadow: 0 4px 24px rgba(139, 92, 246, 0.08) !important;
        }
        
        .stDataFrame thead th {
            background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            font-size: 0.8rem !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            padding: 1rem !important;
            border: none !important;
            text-align: left !important;
        }
        
        .stDataFrame tbody tr {
            border-bottom: 1px solid rgba(139, 92, 246, 0.08) !important;
            transition: all 0.3s ease !important;
        }
        
        .stDataFrame tbody tr:hover {
            background: linear-gradient(90deg, rgba(139, 92, 246, 0.05) 0%, rgba(139, 92, 246, 0.02) 100%) !important;
            transform: scale(1.01) !important;
            box-shadow: 0 2px 8px rgba(139, 92, 246, 0.1) !important;
        }
        
        .stDataFrame tbody td {
            color: var(--text-primary) !important;
            padding: 1rem !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
        }
        
        /* Alternating row colors */
        .stDataFrame tbody tr:nth-child(even) {
            background: rgba(139, 92, 246, 0.02) !important;
        }
        
        /* Enhanced table container */
        .dataframe-container {
            background: white;
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 10px 40px rgba(139, 92, 246, 0.1);
            border: 1px solid rgba(139, 92, 246, 0.1);
            margin: 1rem 0;
        }
        
        /* Custom styled table */
        .modern-table {
            width: 100%;
            background: white;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 4px 24px rgba(139, 92, 246, 0.08);
        }
        
        .modern-table thead {
            background: linear-gradient(135deg, #8B5CF6 0%, #9333EA 100%);
        }
        
        .modern-table thead th {
            color: white;
            font-weight: 600;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 1rem;
            text-align: left;
            border: none;
        }
        
        .modern-table tbody tr {
            border-bottom: 1px solid rgba(139, 92, 246, 0.08);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .modern-table tbody tr:hover {
            background: linear-gradient(90deg, rgba(139, 92, 246, 0.08) 0%, rgba(139, 92, 246, 0.03) 100%);
            transform: translateX(4px);
        }
        
        .modern-table tbody td {
            padding: 1rem;
            color: #374151;
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        /* Status badges in tables */
        .table-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .table-badge.success {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.1));
            color: #059669;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        
        .table-badge.warning {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.1));
            color: #D97706;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
        
        .table-badge.danger {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1));
            color: #DC2626;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        .table-badge.info {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(139, 92, 246, 0.1));
            color: #7C3AED;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }
        
        /* ============================================
           INPUT FIELDS - CLEAN STYLE
           ============================================ */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select {
            background: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 6px !important;
            padding: 0.5rem 0.75rem !important;
            font-size: 0.875rem !important;
            transition: all 0.2s ease !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus {
            border-color: var(--primary-blue) !important;
            box-shadow: 0 0 0 3px rgba(91, 141, 239, 0.1) !important;
            outline: none !important;
        }
        
        /* ============================================
           SIDEBAR - PURPLE THEME
           ============================================ */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(139, 92, 246, 0.02) 0%, rgba(147, 51, 234, 0.01) 100%) !important;
            border-right: 2px solid rgba(139, 92, 246, 0.1) !important;
        }
        
        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.05), rgba(147, 51, 234, 0.02)) !important;
            color: var(--text-primary) !important;
            border: 1px solid rgba(139, 92, 246, 0.2) !important;
            box-shadow: 0 2px 8px rgba(139, 92, 246, 0.1) !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="stSidebar"] .stButton > button:hover {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(147, 51, 234, 0.05)) !important;
            border-color: var(--primary-purple) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.2) !important;
        }
        
        /* Sidebar metrics with purple theme */
        [data-testid="stSidebar"] [data-testid="metric-container"] {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(147, 51, 234, 0.04)) !important;
            border: 1px solid rgba(139, 92, 246, 0.2) !important;
        }
        
        [data-testid="stSidebar"] [data-testid="metric-container"] [data-testid="metric-value"] {
            color: #8B5CF6 !important;
        }
        
        /* Enhanced sidebar input styling */
        [data-testid="stSidebar"] .stNumberInput > div > div > input,
        [data-testid="stSidebar"] .stSelectbox > div > div > select,
        [data-testid="stSidebar"] .stSlider {
            background: rgba(139, 92, 246, 0.05) !important;
            border: 1px solid rgba(139, 92, 246, 0.2) !important;
            border-radius: 8px !important;
        }
        
        [data-testid="stSidebar"] .stNumberInput > div > div > input:focus,
        [data-testid="stSidebar"] .stSelectbox > div > div > select:focus {
            border-color: #8B5CF6 !important;
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1) !important;
        }
        
        /* Purple slider styling for sidebar */
        [data-testid="stSidebar"] .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #8B5CF6, #9333EA) !important;
        }
        
        [data-testid="stSidebar"] .stSlider > div > div > div > div > div {
            background: white !important;
            border: 2px solid #8B5CF6 !important;
            box-shadow: 0 2px 8px rgba(139, 92, 246, 0.3) !important;
        }
        
        /* ============================================
           EXPANDERS - CLEAN STYLE
           ============================================ */
        .streamlit-expander {
            background: var(--bg-card) !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            margin-bottom: 1rem !important;
        }
        
        .streamlit-expanderHeader {
            background: transparent !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-size: 0.875rem !important;
            padding: 0.75rem 1rem !important;
        }
        
        .streamlit-expanderHeader:hover {
            background: var(--bg-tertiary) !important;
        }
        
        /* ============================================
           SPECIAL COMPONENTS - ENHANCED KPIs
           ============================================ */
        
        /* Modern KPI Cards with Gradients */
        .kpi-card {
            background: white;
            border-radius: 20px;
            padding: 1.75rem;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
            border: 1px solid rgba(226, 232, 240, 0.5);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #5B8DEF, #8B5CF6, #5B8DEF);
            background-size: 200% 100%;
            animation: shimmer 3s infinite;
        }
        
        @keyframes shimmer {
            0% { background-position: -200% center; }
            100% { background-position: 200% center; }
        }
        
        /* Advanced Keyframe Animations */
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        
        @keyframes glow {
            0%, 100% { box-shadow: 0 0 20px rgba(139, 92, 246, 0.3); }
            50% { box-shadow: 0 0 40px rgba(139, 92, 246, 0.6), 0 0 60px rgba(139, 92, 246, 0.4); }
        }
        
        @keyframes rotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes slideIn {
            0% { 
                opacity: 0; 
                transform: translateX(-30px); 
            }
            100% { 
                opacity: 1; 
                transform: translateX(0); 
            }
        }
        
        @keyframes scaleIn {
            0% { 
                opacity: 0; 
                transform: scale(0.8); 
            }
            100% { 
                opacity: 1; 
                transform: scale(1); 
            }
        }
        
        @keyframes rainbow {
            0% { filter: hue-rotate(0deg); }
            100% { filter: hue-rotate(360deg); }
        }
        
        @keyframes morphing {
            0%, 100% { border-radius: 20px; }
            25% { border-radius: 30px 20px 25px 15px; }
            50% { border-radius: 15px 25px 20px 30px; }
            75% { border-radius: 25px 15px 30px 20px; }
        }
        
        .kpi-card:hover {
            transform: translateY(-6px) scale(1.02);
            box-shadow: 0 12px 40px rgba(91, 141, 239, 0.15);
        }
        
        /* Enhanced KPI Value with Animation */
        .kpi-value {
            font-size: 2.75rem;
            font-weight: 800;
            background: linear-gradient(135deg, #5B8DEF 0%, #8B5CF6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0.75rem 0;
            font-family: 'Poppins', sans-serif;
            animation: fadeInUp 0.6s ease-out;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .kpi-label {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-family: 'Inter', sans-serif;
        }
        
        .kpi-delta {
            font-size: 0.875rem;
            font-weight: 600;
            padding: 0.375rem 0.75rem;
            border-radius: 20px;
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            margin-top: 0.75rem;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        
        .kpi-delta.positive {
            color: #10B981;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.2));
            border: 1px solid rgba(16, 185, 129, 0.2);
        }
        
        .kpi-delta.negative {
            color: #EF4444;
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.2));
            border: 1px solid rgba(239, 68, 68, 0.2);
        }
        
        /* Premium KPI Card Variants */
        .kpi-card-premium {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: none;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
        }
        
        /* Enhanced Purple Theme for ALL KPIs - Simplified */
        .kpi-card {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.05) 0%, rgba(147, 51, 234, 0.02) 100%) !important;
            border: 1px solid rgba(139, 92, 246, 0.2) !important;
            box-shadow: 0 8px 32px rgba(139, 92, 246, 0.12) !important;
            transition: all 0.3s ease !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .kpi-card::before {
            background: linear-gradient(90deg, #8B5CF6, #9333EA, #7C3AED, #8B5CF6) !important;
            background-size: 300% 100% !important;
            animation: shimmer 3s infinite !important;
        }
        
        .kpi-card:hover {
            box-shadow: 0 16px 48px rgba(139, 92, 246, 0.2) !important;
            border-color: rgba(139, 92, 246, 0.4) !important;
            transform: translateY(-4px) scale(1.02) !important;
        }
        
        .kpi-value {
            background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 50%, #9333EA 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-size: 200% 200% !important;
            animation: gradientShift 3s ease infinite !important;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .kpi-icon {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(147, 51, 234, 0.1)) !important;
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.2) !important;
        }
        
        .kpi-card-blue-gradient {
            background: linear-gradient(135deg, #EBF5FF 0%, #DBEAFE 50%, #BFDBFE 100%);
            border: 1px solid rgba(91, 141, 239, 0.2);
        }
        
        .kpi-card-blue-gradient .kpi-value {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .kpi-card-green-gradient {
            background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 50%, #BBF7D0 100%);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }
        
        .kpi-card-green-gradient .kpi-value {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .kpi-card-purple-gradient {
            background: linear-gradient(135deg, #FAF5FF 0%, #EDE9FE 50%, #DDD6FE 100%);
            border: 1px solid rgba(139, 92, 246, 0.2);
        }
        
        .kpi-card-purple-gradient .kpi-value {
            background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .kpi-card-orange-gradient {
            background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 50%, #FDE68A 100%);
            border: 1px solid rgba(245, 158, 11, 0.2);
        }
        
        .kpi-card-orange-gradient .kpi-value {
            background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Icon container for KPI cards */
        .kpi-icon {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, rgba(91, 141, 239, 0.1), rgba(139, 92, 246, 0.1));
        }
        
        /* Gradient cards for metrics */
        .gradient-card-blue {
            background: linear-gradient(135deg, #667eea 0%, #5B8DEF 100%);
            color: white;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }
        
        .gradient-card-green {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            color: white;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
        }
        
        .gradient-card-orange {
            background: linear-gradient(135deg, #FBBF24 0%, #F59E0B 100%);
            color: white;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(245, 158, 11, 0.3);
        }
        
        .gradient-card-purple {
            background: linear-gradient(135deg, #A78BFA 0%, #8B5CF6 100%);
            color: white;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(139, 92, 246, 0.3);
        }
        
        /* Professional header */
        .professional-header {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.08);
            border: 1px solid rgba(139, 92, 246, 0.2);
            text-align: center;
        }
        
        .professional-header h1 {
            background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0;
        }
        
        .professional-header p {
            color: var(--text-secondary);
            font-size: 1rem;
            margin: 0.5rem 0 0 0;
        }
        
        /* Info boxes */
        .info-box {
            background: linear-gradient(135deg, #EBF5FF 0%, #DBEAFE 100%);
            border-left: 4px solid #5B8DEF;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .success-box {
            background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
            border-left: 4px solid #10B981;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .warning-box {
            background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
            border-left: 4px solid #F59E0B;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .danger-box {
            background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
            border-left: 4px solid #EF4444;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Trading signals */
        .signal-buy {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
            display: inline-block;
        }
        
        .signal-sell {
            background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
            display: inline-block;
        }
        
        /* Enhanced Progress bars */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #8B5CF6, #9333EA, #7C3AED) !important;
            border-radius: 8px !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .stProgress > div > div > div::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
            animation: shimmer 2s infinite;
        }
        
        .stProgress > div > div {
            background: rgba(139, 92, 246, 0.1) !important;
            border-radius: 8px !important;
            border: 1px solid rgba(139, 92, 246, 0.2) !important;
        }
        
        /* Loading spinner */
        .loading-spinner {
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 3px solid rgba(139, 92, 246, 0.3);
            border-radius: 50%;
            border-top-color: #8B5CF6;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Loading overlay */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(139, 92, 246, 0.1);
            backdrop-filter: blur(5px);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }
        
        /* Sliders */
        .stSlider > div > div > div > div {
            background: var(--primary-blue) !important;
        }
        
        .stSlider > div > div > div > div > div {
            background: white !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
        
        /* ============================================
           TOAST NOTIFICATIONS & FEEDBACK
           ============================================ */
        .toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .toast {
            background: white;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
            border-left: 4px solid;
            min-width: 300px;
            animation: slideInRight 0.3s ease-out;
            position: relative;
            overflow: hidden;
        }
        
        .toast::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            animation: progressBar 5s linear;
        }
        
        .toast.success {
            border-left-color: #10B981;
        }
        
        .toast.success::before {
            background: linear-gradient(90deg, #10B981, #059669);
        }
        
        .toast.error {
            border-left-color: #EF4444;
        }
        
        .toast.error::before {
            background: linear-gradient(90deg, #EF4444, #DC2626);
        }
        
        .toast.info {
            border-left-color: #8B5CF6;
        }
        
        .toast.info::before {
            background: linear-gradient(90deg, #8B5CF6, #7C3AED);
        }
        
        .toast.warning {
            border-left-color: #F59E0B;
        }
        
        .toast.warning::before {
            background: linear-gradient(90deg, #F59E0B, #D97706);
        }
        
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(100%);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes progressBar {
            from { width: 100%; }
            to { width: 0%; }
        }
        
        /* Alert enhancements */
        .stAlert {
            border-radius: 12px !important;
            border: none !important;
            backdrop-filter: blur(10px) !important;
        }
        
        .stAlert[data-baseweb="notification"] {
            animation: slideInDown 0.3s ease-out !important;
        }
        
        @keyframes slideInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>
    """

def get_kpi_card_html(value: str, label: str, delta: str = "", color: str = "blue") -> str:
    """Generate HTML for beautiful KPI cards like in the image"""
    gradient_class = f"gradient-card-{color}"
    delta_html = f'<div class="kpi-delta {"positive" if "+" in delta else "negative" if "-" in delta else ""}">{delta}</div>' if delta else ''
    
    return f"""
    <div class="{gradient_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """

def get_metric_card_html(value: str, label: str, sublabel: str = "", icon: str = "") -> str:
    """Generate HTML for clean metric cards"""
    return f"""
    <div class="kpi-card">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                {f'<div style="color: #64748b; font-size: 0.875rem; margin-top: 0.25rem;">{sublabel}</div>' if sublabel else ''}
            </div>
            {f'<div style="font-size: 2rem; color: #cbd5e1;">{icon}</div>' if icon else ''}
        </div>
    </div>
    """

def get_trading_signal_html(signal_type: str, value: str, label: str = "") -> str:
    """Generate HTML for trading signal indicators"""
    signal_class = "signal-buy" if signal_type == "BUY" else "signal-sell"
    icon = "↑" if signal_type == "BUY" else "↓"
    
    return f"""
    <div class="{signal_class}">
        {icon} {label}: <strong>{value}</strong>
    </div>
    """

def get_glass_card_html(title: str, content: str, icon: str = "📊") -> str:
    """Generate HTML for clean cards"""
    return f"""
    <div class="kpi-card">
        <h3 style="color: #1e293b; margin-bottom: 1rem; font-weight: 600;">
            {icon} {title}
        </h3>
        <div style="color: #64748b;">
            {content}
        </div>
    </div>
    """

def get_neon_metric_html(label: str, value: str, delta: str = "", color: str = "#5B8DEF") -> str:
    """Generate HTML for clean styled metrics"""
    delta_html = f'<div class="kpi-delta {"positive" if "+" in delta else "negative"}">{delta}</div>' if delta else ''
    
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color: {color};">{value}</div>
        {delta_html}
    </div>
    """

def get_enhanced_kpi_card(value: str, label: str, delta: str = "", icon: str = "📊", color: str = "blue") -> str:
    """Generate enhanced KPI card with gradient background and animations"""
    delta_html = ""
    if delta:
        delta_class = "positive" if "+" in delta else "negative" if "-" in delta else ""
        arrow = "↑" if "+" in delta else "↓" if "-" in delta else ""
        delta_html = f'<div class="kpi-delta {delta_class}">{arrow} {delta}</div>'
    
    return f"""
    <div class="kpi-card kpi-card-{color}-gradient">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """

def get_modern_table_html(df, title: str = "", show_index: bool = False) -> str:
    """Generate a modern styled HTML table from a pandas DataFrame"""
    html = f"""
    <div class="dataframe-container">
        {f'<h3 style="color: #8B5CF6; margin-bottom: 1rem; font-weight: 600;">{title}</h3>' if title else ''}
        <table class="modern-table">
            <thead>
                <tr>
    """
    
    # Add index column if needed
    if show_index:
        html += '<th style="text-align: center;">Index</th>'
    
    # Add column headers
    for col in df.columns:
        html += f'<th>{col}</th>'
    
    html += """
                </tr>
            </thead>
            <tbody>
    """
    
    # Add data rows
    for idx, row in df.iterrows():
        html += '<tr>'
        if show_index:
            html += f'<td style="text-align: center; font-weight: 600;">{idx}</td>'
        
        for col in df.columns:
            value = row[col]
            # Format different data types
            if isinstance(value, (int, float)):
                if 'P&L' in col or '$' in col or 'Value' in col:
                    formatted_value = f'${value:,.2f}' if value >= 0 else f'-${abs(value):,.2f}'
                    color = '#10B981' if value >= 0 else '#EF4444'
                    html += f'<td style="color: {color}; font-weight: 600;">{formatted_value}</td>'
                elif '%' in col or 'Rate' in col or 'Score' in col:
                    formatted_value = f'{value:.2f}%' if 'Score' not in col else f'{value:.1f}'
                    html += f'<td>{formatted_value}</td>'
                else:
                    html += f'<td>{value:.4f}</td>'
            elif 'Status' in col:
                # Add status badge
                badge_class = 'success' if 'Profitable' in str(value) else 'warning' if 'Monitor' in str(value) else 'info'
                html += f'<td><span class="table-badge {badge_class}">{value}</span></td>'
            else:
                html += f'<td>{value}</td>'
        
        html += '</tr>'
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    return html

def get_tooltip_html(content: str, tooltip_text: str) -> str:
    """Generate HTML with tooltip functionality"""
    tooltip_id = f"tooltip-{hash(tooltip_text) % 10000}"
    
    return f"""
    <div class="tooltip-container-{tooltip_id}" style="position: relative; display: inline-block;">
        {content}
        <div class="tooltip-text-{tooltip_id}" style="
            visibility: hidden;
            width: 200px;
            background: linear-gradient(135deg, rgb(31, 41, 55) 0%, rgb(55, 65, 81) 100%);
            color: white;
            text-align: center;
            border-radius: 8px;
            padding: 8px 12px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.875rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        ">
            {tooltip_text}
            <div style="
                position: absolute;
                top: 100%;
                left: 50%;
                margin-left: -5px;
                border-width: 5px;
                border-style: solid;
                border-color: rgb(55, 65, 81) transparent transparent transparent;
            "></div>
        </div>
    </div>
    <style>
        .tooltip-container-{tooltip_id}:hover .tooltip-text-{tooltip_id} {{
            visibility: visible !important;
            opacity: 1 !important;
        }}
    </style>
    """

def get_loading_spinner(text: str = "Loading...") -> str:
    """Generate loading spinner HTML"""
    return f"""
    <div style="display: flex; align-items: center; justify-content: center; padding: 2rem;">
        <div class="loading-spinner"></div>
        <span style="margin-left: 1rem; color: #8B5CF6; font-weight: 600;">{text}</span>
    </div>
    """

def get_success_toast(message: str) -> str:
    """Generate success toast notification"""
    return f"""
    <div class="toast success">
        <div style="display: flex; align-items: center;">
            <div style="color: #10B981; font-size: 1.25rem; margin-right: 0.75rem;">✅</div>
            <div>
                <div style="font-weight: 600; color: #065f46;">Success!</div>
                <div style="color: #6b7280; font-size: 0.875rem;">{message}</div>
            </div>
        </div>
    </div>
    """

def get_error_toast(message: str) -> str:
    """Generate error toast notification"""
    return f"""
    <div class="toast error">
        <div style="display: flex; align-items: center;">
            <div style="color: #EF4444; font-size: 1.25rem; margin-right: 0.75rem;">❌</div>
            <div>
                <div style="font-weight: 600; color: #7f1d1d;">Error!</div>
                <div style="color: #6b7280; font-size: 0.875rem;">{message}</div>
            </div>
        </div>
    </div>
    """

def get_info_card(title: str, content: str, icon: str = "ℹ️") -> str:
    """Generate informational card with icon"""
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.05) 0%, rgba(147, 51, 234, 0.02) 100%);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.1);
    ">
        <div style="display: flex; align-items: flex-start;">
            <div style="font-size: 1.5rem; margin-right: 1rem;">{icon}</div>
            <div>
                <h4 style="color: #8B5CF6; margin: 0 0 0.5rem 0; font-weight: 600;">{title}</h4>
                <p style="color: #6b7280; margin: 0; line-height: 1.5;">{content}</p>
            </div>
        </div>
    </div>
    """

def get_premium_metric_card(value: str, label: str, sublabel: str = "", trend_data: list = None) -> str:
    """Generate premium metric card with mini chart"""
    trend_html = ""
    if trend_data:
        # Create simple sparkline
        max_val = max(trend_data) if trend_data else 1
        min_val = min(trend_data) if trend_data else 0
        range_val = max_val - min_val if max_val != min_val else 1
        
        points = []
        for i, val in enumerate(trend_data):
            x = (i / (len(trend_data) - 1)) * 100 if len(trend_data) > 1 else 50
            y = 50 - ((val - min_val) / range_val) * 40 if range_val > 0 else 30
            points.append(f"{x},{y}")
        
        trend_html = f"""
        <svg width="100" height="50" style="margin-top: 0.5rem;">
            <polyline points="{' '.join(points)}" 
                      fill="none" 
                      stroke="url(#gradient)" 
                      stroke-width="2"/>
            <defs>
                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#5B8DEF;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#8B5CF6;stop-opacity:1" />
                </linearGradient>
            </defs>
        </svg>
        """
    
    return f"""
    <div class="kpi-card kpi-card-premium">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {f'<div style="color: #94a3b8; font-size: 0.875rem; margin-top: 0.25rem;">{sublabel}</div>' if sublabel else ''}
        {trend_html}
    </div>
    """


# ========== PROFESSIONAL KPI & VISUAL COMPONENTS ==========
# Based on Sky Central's advanced design patterns

def create_kpi_card(value: str, label: str, icon: str = "📊", color: str = "#0052CC", height: str = "auto", additional_info: str = "") -> str:
    """
    Create a professional KPI card with gradient background and glassmorphism
    
    Args:
        value: Main metric value (e.g., "94.2%" or "$1.2M")
        label: Primary label
        icon: Emoji or icon
        color: Primary color (hex)
        height: CSS height value
        additional_info: Optional additional text at bottom
    
    Returns:
        HTML string for KPI card
    """
    info_box = f'''
    <div style="background: rgba(255,255,255,0.95); padding: 0.5rem; border-radius: 8px; 
                color: #1f2937; margin-top: 0.75rem;">
        <p style="margin: 0; font-size: 0.85rem; font-weight: 600;">{additional_info}</p>
    </div>
    ''' if additional_info else ''
    
    return f"""
    <div style="background: linear-gradient(145deg, {color} 0%, {color}dd 100%); 
                padding: 2rem 1.5rem; border-radius: 20px; text-align: center; color: white;
                box-shadow: 0 10px 30px {color}40; margin-bottom: 1rem;
                border: 1px solid rgba(255, 255, 255, 0.2); position: relative; height: {height}; 
                display: flex; flex-direction: column; justify-content: space-between;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);"
         onmouseover="this.style.transform='translateY(-4px)'; this.style.boxShadow='0 20px 45px {color}50';"
         onmouseout="this.style.transform=''; this.style.boxShadow='0 10px 30px {color}40';">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div>
            <h2 style="font-size: 2.3rem; margin: 0.5rem 0 0 0; font-weight: 800; 
                       text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                {value}
            </h2>
            <p style="margin: 0.6rem 0 0 0; font-size: 0.95rem; opacity: 0.95; font-weight: 600;">
                {label}
            </p>
        </div>
        {info_box}
    </div>
    """


def create_ios_glass_stat_card(value: str, label: str, icon: str = "", trend: dict = None, variant: str = 'blue') -> str:
    """
    Create an iOS-style glassmorphism stat card with optional trend indicator
    
    Args:
        value: Main statistic value (e.g., "$1.2M" or "1,234")
        label: Description label
        icon: Optional emoji or icon
        trend: Optional dict with 'direction' ('up'/'down') and 'value' (e.g., '12%')
        variant: Color variant ('blue', 'green', 'orange', 'purple', 'red')
    
    Returns:
        HTML string for stat card
    """
    variant_colors = {
        'blue': {'primary': 'rgb(59, 130, 246)', 'light': 'rgba(59, 130, 246, 0.15)'},
        'green': {'primary': 'rgb(16, 185, 129)', 'light': 'rgba(16, 185, 129, 0.15)'},
        'orange': {'primary': 'rgb(249, 115, 22)', 'light': 'rgba(249, 115, 22, 0.15)'},
        'purple': {'primary': 'rgb(168, 85, 247)', 'light': 'rgba(168, 85, 247, 0.15)'},
        'red': {'primary': 'rgb(239, 68, 68)', 'light': 'rgba(239, 68, 68, 0.15)'}
    }
    
    color = variant_colors.get(variant, variant_colors['blue'])
    icon_html = f'<div style="font-size: 2rem; margin-bottom: 0.75rem;">{icon}</div>' if icon else ''
    
    trend_html = ''
    if trend:
        trend_color = 'rgb(16, 185, 129)' if trend['direction'] == 'up' else 'rgb(239, 68, 68)'
        trend_icon = '↑' if trend['direction'] == 'up' else '↓'
        trend_html = f'''
        <div style="display: inline-flex; align-items: center; gap: 0.25rem; 
                    padding: 0.25rem 0.75rem; border-radius: 9999px;
                    background: rgba(255, 255, 255, 0.4);
                    backdrop-filter: blur(12px);
                    -webkit-backdrop-filter: blur(12px);
                    margin-left: 0.5rem;">
            <span style="color: {trend_color}; font-weight: 700; font-size: 0.875rem;">{trend_icon}</span>
            <span style="color: {trend_color}; font-weight: 600; font-size: 0.8rem;">{trend['value']}</span>
        </div>
        '''
    
    html = f"""<div style="background: linear-gradient(135deg, {color['light']} 0%, rgba(255, 255, 255, 0.7) 100%); backdrop-filter: blur(40px) saturate(180%); -webkit-backdrop-filter: blur(40px) saturate(180%); border: 1px solid rgba(255, 255, 255, 0.5); border-radius: 1.25rem; padding: 1.5rem; margin: 0.75rem 0; box-shadow: 0 8px 32px rgba(31, 38, 135, 0.12), 0 2px 8px rgba(31, 38, 135, 0.06), inset 0 1px 2px rgba(255, 255, 255, 0.9); transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); position: relative; overflow: hidden; text-align: center;" onmouseover="this.style.transform='translateY(-6px) scale(1.02)'; this.style.boxShadow='0 20px 60px rgba(31, 38, 135, 0.15), 0 8px 32px rgba(31, 38, 135, 0.08), inset 0 1px 2px rgba(255, 255, 255, 0.95)';" onmouseout="this.style.transform=''; this.style.boxShadow='0 8px 32px rgba(31, 38, 135, 0.12), 0 2px 8px rgba(31, 38, 135, 0.06), inset 0 1px 2px rgba(255, 255, 255, 0.9)';"><div style="position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent 0%, {color['primary']}80 50%, transparent 100%);"></div>{icon_html}<div style="display: flex; align-items: baseline; justify-content: center; margin-bottom: 0.5rem;"><div style="font-size: 2.25rem; font-weight: 800; color: {color['primary']}; text-shadow: 0 2px 12px {color['light']};">{value}</div>{trend_html}</div><div style="font-size: 0.875rem; font-weight: 600; color: hsl(217, 33%, 17%); text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.8;">{label}</div></div>"""
    return html


def create_page_header(title: str, subtitle: str, icon: str = "📊") -> str:
    """
    Create a modern, professional page header with gradient text
    
    Args:
        title: Main page title
        subtitle: Subtitle or description
        icon: Emoji or icon
    
    Returns:
        HTML string for page header
    """
    return f"""<div style="text-align: center; margin: 2.5rem 0 2rem 0;"><h2 style="background: linear-gradient(135deg, #1e293b 0%, #475569 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 2.5rem; font-weight: 800; letter-spacing: -0.04em; margin: 0; line-height: 1.1;">{icon} {title}</h2><p style="color: #64748b; font-size: 1.125rem; font-weight: 400; margin: 0.75rem 0 0 0; letter-spacing: -0.01em;">{subtitle}</p></div>"""


def create_info_section(title: str, content: str, icon: str = "💡", color: str = "#0052CC") -> str:
    """
    Create an informational section with gradient background
    
    Args:
        title: Section title
        content: Section content/description
        icon: Emoji or icon
        color: Primary color (hex)
    
    Returns:
        HTML string for info section
    """
    return f"""<div style="background: linear-gradient(135deg, {color}15 0%, {color}08 100%); border-radius: 16px; padding: 1.5rem; margin: 1.5rem 0; border-left: 6px solid {color}; box-shadow: 0 8px 24px {color}15;"><div style="display: flex; align-items: center; gap: 0.75rem;"><span style="font-size: 1.5rem;">{icon}</span><div><h4 style="margin: 0; color: {color}; font-weight: 600;">{title}</h4><p style="margin: 0.5rem 0 0 0; color: #475569; font-size: 0.9rem;">{content}</p></div></div></div>"""


def create_status_badge(count: int, status: str, color: str, description: str = "", icon: str = "", height: str = "auto") -> str:
    """
    Create a status badge card with count and description
    
    Args:
        count: Numeric count
        status: Status label
        color: Badge color (hex)
        description: Optional description text
        icon: Optional emoji or icon
        height: CSS height value
    
    Returns:
        HTML string for status badge
    """
    desc_html = f'<div style="font-size: 0.8rem; color: #6b7280; line-height: 1.3; margin-top: 0.5rem;">{description}</div>' if description else ''
    icon_html = f'<div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>' if icon else ''
    
    return f"""
    <div style="background: linear-gradient(145deg, {color}15, {color}08); 
                padding: 1.5rem; border-radius: 15px; text-align: center;
                border: 2px solid {color}30; margin-bottom: 1rem; height: {height}; 
                display: flex; flex-direction: column; justify-content: center;
                transition: all 0.3s ease;"
         onmouseover="this.style.transform='scale(1.03)'; this.style.borderColor='{color}60';"
         onmouseout="this.style.transform=''; this.style.borderColor='{color}30';">
        {icon_html}
        <div style="font-size: 2rem; font-weight: 700; color: {color}; margin-bottom: 0.5rem;">{count}</div>
        <div style="font-size: 0.9rem; font-weight: 600; color: #374151; margin-bottom: 0.5rem;">{status}</div>
        {desc_html}
    </div>
    """


def create_warning_banner(title: str, message: str, icon: str = "⚠️", color: str = "#f59e0b") -> str:
    """
    Create a warning or alert banner
    
    Args:
        title: Warning title
        message: Warning message
        icon: Emoji or icon
        color: Warning color (hex)
    
    Returns:
        HTML string for warning banner
    """
    return f"""
    <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                padding: 2rem; border-radius: 15px; text-align: center; color: white; margin: 2rem 0;
                box-shadow: 0 8px 25px {color}40;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
        <h3 style="margin: 0; font-size: 1.5rem; font-weight: 700;">{title}</h3>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1rem;">{message}</p>
    </div>
    """


def create_glossy_section_header(icon: str, title: str, subtitle: str) -> str:
    """
    Create a glossy section header with icon, title, and subtitle
    
    Args:
        icon: Emoji or icon
        title: Section title
        subtitle: Section subtitle
    
    Returns:
        HTML string for section header
    """
    return f"""<div style="background: rgba(0, 82, 204, 0.05); backdrop-filter: blur(24px) saturate(180%); -webkit-backdrop-filter: blur(24px) saturate(180%); border: 1px solid rgba(0, 82, 204, 0.2); border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); transition: box-shadow 0.3s ease;" onmouseover="this.style.boxShadow='0 4px 12px 0 rgba(0, 82, 204, 0.15)';" onmouseout="this.style.boxShadow='0 1px 3px 0 rgba(0, 0, 0, 0.1)';"><div style="display: flex; align-items: center; gap: 1rem;"><div style="padding: 0.75rem; border-radius: 0.75rem; background: rgba(0, 82, 204, 0.1); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); display: flex; align-items: center; justify-content: center;"><span style="font-size: 1.5rem;">{icon}</span></div><div style="flex: 1;"><h2 style="color: hsl(217, 33%, 17%); font-size: 1.25rem; font-weight: 700; margin: 0 0 0.25rem 0; line-height: 1.2;">{title}</h2><p style="color: hsl(215, 16%, 47%); font-size: 0.875rem; font-weight: 400; margin: 0; line-height: 1.25;">{subtitle}</p></div></div></div>"""
