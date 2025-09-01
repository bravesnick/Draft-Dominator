import streamlit as st
import pandas as pd
import numpy as np
import uuid

# --- App Version ---
APP_VERSION = "5.5 Stable"

# --- Page Configuration ---
st.set_page_config(
    page_title="Ultimate Fantasy Football Draft Dominator",
    page_icon="🏈",
    layout="wide"
)

def load_custom_css():
    """Applies custom CSS for a better dark theme experience."""
    st.markdown("""
        <style>
            /* Custom styling for DataFrame headers in dark mode */
            .stDataFrame thead th {
                background-color: #2E2E2E !important;
                color: #FAFAFA !important;
            }
            /* Style for the main action buttons */
            .stButton > button {
                border: 1px solid #FFC300 !important;
                color: #FFC300 !important;
                background-color: transparent;
            }
            .stButton > button:hover {
                background-color: #FFC300 !important;
                color: #1E1E1E !important;
            }
            /* Make the main 'Draft' button in the recommendation section stand out */
            div[data-testid="stHorizontalBlock"] .stButton > button {
                 background-color: #FFC300 !important;
                 color: #1E1E1E !important;
            }
            /* Center text in all dataframe cells */
            .stDataFrame td, .stDataFrame th {
                text-align: center !important;
            }
        </style>
    """, unsafe_allow_html=True)

# --- App State Initialization ---
def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    defaults = {
        'draft_started': False, 'projections_df': None, 'draft_history': [],
        'current_pick': 1, 'team_count': 10, 'draft_position': 1,
        'roster': {pos: [] for pos in ['QB', 'RB', 'WR', 'TE', 'W/R', 'K', 'DEF', 'BENCH']},
        'player_search': ""
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

# --- Data Processing Functions ---
@st.cache_data
def load_and_process_data(uploaded_file):
    """Loads, cleans, and processes player projection data."""
    try:
        df = pd.read_csv(uploaded_file)
        required = ['Player', 'Pos', 'Team', 'ADP', 'ProjectedPoints']
        if not all(col in df.columns for col in required):
            st.error(f"CSV must contain columns: {', '.join(required)}")
            return None
        
        df['ADP'] = pd.to_numeric(df['ADP'], errors='coerce').fillna(999)
        df['ProjectedPoints'] = pd.to_numeric(df['ProjectedPoints'], errors='coerce').fillna(0)
        df = df.sort_values('ProjectedPoints', ascending=False).drop_duplicates('Player', keep='first')
        
        # VORP Calculation
        df['VORP'] = 0.0
        for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
            pos_df = df[df['Pos'] == pos]
            if not pos_df.empty:
                level_map = {'QB': 12, 'RB': 30, 'WR': 35, 'TE': 12, 'K': 10, 'DEF': 10}
                rep_index = min(level_map.get(pos, 0), len(pos_df) - 1)
                rep_level = pos_df['ProjectedPoints'].nlargest(rep_index + 1).iloc[-1]
                df.loc[df['Pos'] == pos, 'VORP'] = df['ProjectedPoints'] - rep_level

        df['FormattedName'] = df['Player'].apply(lambda x: f"{x.split()[0][0]}. {' '.join(x.split()[1:])}" if isinstance(x, str) and len(x.split()) > 1 else x)
        df = df.sort_values('VORP', ascending=False).reset_index(drop=True)
        df['UniqueKey'] = df.index
        if 'Bye' not in df.columns: df['Bye'] = np.random.randint(5, 15, size=len(df))
        if 'Risk' not in df.columns: df['Risk'] = 'Low'
        return df
    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
        return None

# --- Roster and Draft Logic ---
def add_to_roster(player, roster):
    pos = player['Pos']
    limits = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'W/R': 1, 'K': 1, 'DEF': 1, 'BENCH': 6}
    
    if pos in ['QB', 'TE', 'K', 'DEF'] and len(roster[pos]) < limits[pos]:
        roster[pos].append(player)
    elif pos in ['RB', 'WR'] and len(roster[pos]) < limits[pos]:
        roster[pos].append(player)
    elif pos in ['RB', 'WR', 'TE'] and len(roster['W/R']) < limits['W/R']:
        roster['W/R'].append(player)
    elif len(roster['BENCH']) < limits['BENCH']:
        roster['BENCH'].append(player)
    else:
        st.toast(f"Roster full. Could not add {player['Player']}.", icon="⚠️")
    return roster


def is_roster_full(roster):
    """Checks if the roster is full."""
    structure = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'W/R': 1, 'K': 1, 'DEF': 1, 'BENCH': 6}
    return all(len(roster.get(pos, [])) >= limit for pos, limit in structure.items())

def handle_draft_action(player_key, action):
    player_series = st.session_state.projections_df.loc[st.session_state.projections_df['UniqueKey'] == player_key]
    if player_series.empty: return
    player = player_series.iloc[0].to_dict()
    
    st.session_state.draft_history.append({'Player': player['Player'], 'drafted_by': action, 'Pick': st.session_state.current_pick})
    if action == 'me':
        st.session_state.roster = add_to_roster(player, st.session_state.roster)
    st.session_state.current_pick += 1
    st.session_state.player_search = "" # Clear search after action

def undo_last_pick():
    if st.session_state.draft_history:
        last_action = st.session_state.draft_history.pop()
        if last_action['drafted_by'] == 'me':
            st.session_state.roster = {pos: [p for p in players if p['Player'] != last_action['Player']] for pos, players in st.session_state.roster.items()}
        st.session_state.current_pick -= 1

def handle_other_taken_action():
    st.session_state.draft_history.append({'Player': 'Player Not In CSV', 'drafted_by': 'opponent', 'Pick': st.session_state.current_pick})
    st.session_state.current_pick += 1
    st.session_state.player_search = "" # Clear the search box

# --- Expert Strategy & Player Lists ---
HIGH_UPSIDE_PLAYERS = {'QB': ['Anthony Richardson', 'Jayden Daniels', 'Josh Allen'], 'RB': ['Zamir White', 'Tyjae Spears', 'Jaylen Wright', "De'Von Achane"], 'WR': ['George Pickens', 'Jameson Williams', 'Ladd McConkey', 'Tank Dell'], 'TE': ['Kyle Pitts', 'Jake Ferguson', 'Brock Bowers']}
ROOKIE_GEMS = ['Marvin Harrison Jr.', 'Malik Nabers', 'Rome Odunze', 'Jonathan Brooks', 'Trey Benson', 'Caleb Williams']
SLEEPER_PICKS = {'RB': ['Tyjae Spears', 'Jaylen Wright', 'Chase Brown'], 'WR': ['Khalil Shakir', 'Rashid Shaheed', 'Demario Douglas'], 'TE': ['Luke Musgrave', 'Isaiah Likely']}

def get_recommendations(roster, available_players, current_pick, team_count):
    if available_players.empty: return None
    scored_players = available_players.copy()
    round_num = (current_pick - 1) // team_count + 1
    
    scored_players['value_score'] = scored_players['VORP']
    scarcity_bonuses = {pos: max(0, 10 - (len(scored_players[scored_players['Pos'] == pos]) * 0.5)) for pos in ['QB', 'RB', 'WR', 'TE']}
    scored_players['scarcity_bonus'] = scored_players['Pos'].map(scarcity_bonuses).fillna(0)
    
    total_positions = {pos: len(roster.get(pos, [])) + sum(1 for p in roster.get('BENCH', []) if p.get('Pos') == pos) for pos in ['QB', 'TE']}
    needs = {'QB': 20, 'RB': 25, 'WR': 25, 'TE': 15, 'W/R': 15, 'K': 10, 'DEF': 10}
    
    def get_need_bonus(player):
        pos = player['Pos']
        starting_filled = all(len(roster.get(p, [])) >= (2 if p in ['RB', 'WR'] else 1) for p in ['QB', 'RB', 'WR', 'TE', 'W/R'])
        if starting_filled and pos in ['K', 'DEF'] and not roster.get(pos, []): return 50
        if (pos in ['K', 'DEF'] and roster.get(pos, [])) or (pos == 'QB' and total_positions.get('QB', 0) >= 2) or (pos == 'TE' and total_positions.get('TE', 0) >= 2) or len(roster.get('BENCH', [])) >= 6: return -9999
        bonus = needs.get(pos, 0) if not roster.get(pos, []) else 0
        if round_num > 9 and pos in ['QB', 'TE'] and total_positions.get(pos, 0) == 1: bonus += 12
        return bonus
    scored_players['need_bonus'] = scored_players.apply(get_need_bonus, axis=1)

    scored_players['special_bonus'] = scored_players.apply(lambda p: sum([
        7 if p['Player'] in HIGH_UPSIDE_PLAYERS.get(p.get('Pos', ''), []) else 0,
        5 if p['Player'] in ROOKIE_GEMS else 0,
        8 if p['Player'] in SLEEPER_PICKS.get(p.get('Pos', ''), []) else 0
    ]), axis=1)

    scored_players['adp_value'] = (scored_players['ADP'] - current_pick).clip(lower=0) * 0.1
    
    bye_counts = {}
    for pos_list in roster.values():
        for p in pos_list:
            bye_counts[p['Bye']] = bye_counts.get(p['Bye'], 0) + 1
    scored_players['bye_penalty'] = scored_players['Bye'].apply(lambda bye: -5 * bye_counts.get(bye, 0) if bye_counts.get(bye, 0) > 2 else 0)

    weights = {'value': 1.0, 'scarcity': 1.2, 'need': 1.5, 'special': 1.0, 'adp': 1.2, 'bye': 1.2} if 4 < round_num <= 9 else {'value': 1.0, 'scarcity': 1.0, 'need': 2.0, 'special': 1.5, 'adp': 1.5, 'bye': 1.5} if round_num > 9 else {'value': 1.0, 'scarcity': 1.0, 'need': 1.0, 'special': 0.5, 'adp': 1.0, 'bye': 1.0}
    
    scored_players['total_score'] = sum(weights.get(col.split('_')[0], 0) * scored_players[col] for col in ['value_score', 'scarcity_bonus', 'need_bonus', 'special_bonus', 'adp_value', 'bye_penalty'])

    rec_player = scored_players.sort_values('total_score', ascending=False).iloc[0]
    return {'player': rec_player.to_dict(), 'reason': f"Top pick (VORP: {rec_player['VORP']:.1f}, ADP: {rec_player['ADP']:.1f})"}

# --- Team Analysis & Export ---
def analyze_team(roster, projections_df):
    st.subheader("🏆 Final Roster & Draft Analysis")
    
    def get_player_grade(player, projections_df):
        if player is None: return "-"
        pos = player.get('Pos')
        if not pos: return "N/A"
        vorp = player.get('VORP', 0)
        
        pos_vorps = projections_df[projections_df['Pos'] == pos]['VORP']
        
        if pos_vorps.empty or pos_vorps.max() <= 0:
            pos_points = projections_df[projections_df['Pos'] == pos]['ProjectedPoints']
            if pos_points.empty: return "C"
            if player['ProjectedPoints'] > pos_points.quantile(0.75): return "A"
            elif player['ProjectedPoints'] > pos_points.quantile(0.5): return "B"
            else: return "C"
            
        percentile = (vorp / pos_vorps.max()) * 100 if pos_vorps.max() > 0 else 0
        if percentile >= 90: return "A+"
        if percentile >= 80: return "A"
        if percentile >= 70: return "B+"
        if percentile >= 60: return "B"
        if percentile >= 50: return "C+"
        if percentile >= 40: return "C"
        return "D"

    roster_data = []
    structure = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'W/R': 1, 'K': 1, 'DEF': 1, 'BENCH': 6}
    
    for pos, limit in structure.items():
        players = roster.get(pos, [])
        for i in range(limit):
            player = players[i] if i < len(players) else None
            player_display = "-"
            if player:
                indicator = ""
                if player['Player'] in HIGH_UPSIDE_PLAYERS.get(player.get('Pos', ''), []): indicator += "🔥"
                if player['Player'] in ROOKIE_GEMS: indicator += "💎"
                if player['Player'] in SLEEPER_PICKS.get(player.get('Pos', ''), []): indicator += "🕵️"
                
                player_display = f"{player['FormattedName']} {indicator}".strip()
                if pos == 'BENCH':
                    player_display += f" ({player['Pos']})"

            roster_data.append({
                "Position": pos,
                "Player": player_display,
                "Team": player['Team'] if player else "-",
                "VORP": f"{player['VORP']:.1f}" if player else "-",
                "Bye Week": player['Bye'] if player else "-",
                "Grade": get_player_grade(player, projections_df)
            })

    num_rows = len(roster_data)
    df_height = (num_rows + 1) * 35 + 3
    st.dataframe(pd.DataFrame(roster_data), use_container_width=True, hide_index=True, height=df_height)
    
    export_roster()

def export_roster():
    """Provides a button to download the final roster as a CSV."""
    roster_data = []
    for pos, players in st.session_state.roster.items():
        for player in players:
            roster_data.append({
                'Position': pos,
                'Player': player['Player'],
                'Team': player['Team'],
                'ProjectedPoints': player['ProjectedPoints'],
                'VORP': player['VORP'],
                'Bye': player['Bye']
            })
    if roster_data:
        roster_df = pd.DataFrame(roster_data)
        csv = roster_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Roster as CSV",
            data=csv,
            file_name='final_draft_roster.csv',
            mime='text/csv',
        )

# --- UI Display Functions ---
def show_setup_screen():
    st.header(f"🏈 Ultimate Fantasy Football Draft Dominator v{APP_VERSION}")
    st.markdown("Upload your player projections CSV (must include Player, Pos, Team, ADP, ProjectedPoints).")
    uploaded_file = st.file_uploader("Upload Projections CSV", type=["csv"])
    if uploaded_file:
        projections_df = load_and_process_data(uploaded_file)
        if projections_df is not None:
            st.session_state.projections_df = projections_df
            st.success("Projections loaded! Configure your league below.")
            col1, col2 = st.columns(2)
            st.session_state.team_count = col1.number_input("Teams in League", 8, 16, 10, 1)
            st.session_state.draft_position = col2.number_input("Your Draft Position", 1, st.session_state.team_count, 1, 1)
            if st.button("Start Draft", type="primary", use_container_width=True):
                st.session_state.draft_started = True
                st.rerun()

def display_draft_board_and_controls():
    st.subheader("📋 Draft Board")
    drafted = [p['Player'] for p in st.session_state.draft_history]
    available = st.session_state.projections_df[~st.session_state.projections_df['Player'].isin(drafted)]
    
    st.text_input("🔍 Search for a player (Global Search)", key="player_search")
    query = st.session_state.player_search

    st.button("Player Taken (Not in List)", on_click=handle_other_taken_action, use_container_width=True)

    results = available[available['Player'].str.contains(query, case=False, na=False)] if query else available
    
    if query and results.empty:
        st.warning(f"No available players found for '{query}'.")

    tabs = st.tabs(["⭐ Best", "QB", "RB", "WR", "TE", "DEF/K", "📜 History"])
    pos_map = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "DEF/K": ["DEF", "K"]}

    for i, title in enumerate(["⭐ Best", "QB", "RB", "WR", "TE", "DEF/K"]):
        with tabs[i]:
            df = results[results.Pos.isin(pos_map[title])] if title in pos_map and isinstance(pos_map[title], list) else (results[results.Pos == pos_map[title]] if title in pos_map else results)
            df_view = df.sort_values('VORP', ascending=False).head(30)
            if not df_view.empty:
                cols = st.columns((3, 1, 1, 1, 1, 0.8, 0.8))
                headers = ['Player', 'Pos', 'Team', 'VORP', 'ADP', 'Me', 'Opp']
                for col, header in zip(cols, headers): col.markdown(f"**{header}**")
                for _, p_row in df_view.iterrows():
                    cols = st.columns((3, 1, 1, 1, 1, 0.8, 0.8))
                    p = p_row.to_dict()
                    cols[0].write(p['FormattedName'])
                    cols[1].write(p['Pos'])
                    cols[2].write(p['Team'])
                    cols[3].write(f"{p['VORP']:.1f}")
                    cols[4].write(f"{p['ADP']:.1f}")
                    cols[5].button("🏆", key=f"draft_me_{title}_{p['UniqueKey']}", on_click=handle_draft_action, args=(p['UniqueKey'], 'me'))
                    cols[6].button("❌", key=f"draft_opp_{title}_{p['UniqueKey']}", on_click=handle_draft_action, args=(p['UniqueKey'], 'opponent'))
            elif not query: st.write("No players available.")
    with tabs[6]:
        if st.session_state.draft_history:
            st.dataframe(pd.DataFrame(st.session_state.draft_history).sort_index(ascending=False), use_container_width=True, hide_index=True)

def display_my_team():
    st.subheader("🏆 Your Team")
    roster_df_data = []
    structure = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'W/R': 1, 'K': 1, 'DEF': 1, 'BENCH': 6}
    for pos, limit in structure.items():
        players = st.session_state.roster.get(pos, [])
        for i in range(limit):
            p = players[i] if i < len(players) else None
            display = f"{p['FormattedName']} (Bye: {p['Bye']})" + (f" ({p['Pos']})" if pos == 'BENCH' else "") if p else "-"
            roster_df_data.append({"Pos": pos, "Player": display, "VORP": p['VORP'] if p else 0})
    st.dataframe(pd.DataFrame(roster_df_data), use_container_width=True, hide_index=True, height=565)

def get_my_picks():
    team_count, draft_pos = st.session_state.team_count, st.session_state.draft_position
    return [r * team_count + (draft_pos if r % 2 == 0 else team_count - draft_pos + 1) for r in range(15)]

def display_drafting_screen():
    st.header(f"Draft Dominator v{APP_VERSION} - Round {(st.session_state.current_pick - 1) // st.session_state.team_count + 1}, Pick {st.session_state.current_pick}")
    
    if is_roster_full(st.session_state.roster):
        st.success("🎉 Draft Complete!")
        analyze_team(st.session_state.roster, st.session_state.projections_df)
        return

    if st.session_state.draft_history:
        st.button("Undo Last Pick", on_click=undo_last_pick, use_container_width=True)

    st.divider()
    st.subheader("💡 Expert Recommendation")
    available = st.session_state.projections_df[~st.session_state.projections_df['Player'].isin([p['Player'] for p in st.session_state.draft_history])]

    if st.session_state.current_pick in get_my_picks():
        st.markdown("""<div style="background-color: #2E2E2E; border-left: 6px solid #FFC300; border-radius: 5px; padding: 10px; color: #FFC300; font-weight: bold; margin-bottom: 1rem;">🔥 YOUR PICK IS IN! Here's your top option: 🔥</div>""", unsafe_allow_html=True)

    if not available.empty:
        rec = get_recommendations(st.session_state.roster, available, st.session_state.current_pick, st.session_state.team_count)
        if rec:
            player = rec['player']
            col1, col2 = st.columns([4, 1.5])
            col1.markdown(f"##### ✅ **{player['FormattedName']} ({player['Team']})**")
            col1.markdown(f"> _{rec['reason']}_")
            col2.button("🏆 Draft", on_click=handle_draft_action, args=(player['UniqueKey'], 'me'), use_container_width=True, key=f"draft_rec_{player['UniqueKey']}")

    st.divider()
    draft_col, team_col = st.columns([3, 2])
    with draft_col: display_draft_board_and_controls()
    with team_col: display_my_team()

# --- Main App Logic ---
initialize_session_state()
load_custom_css()

if not st.session_state.draft_started:
    show_setup_screen()
else:
    if st.session_state.projections_df is not None:
        display_drafting_screen()
    else:
        st.error("Player data not found. Please restart.")
        if st.button("Go to Setup", on_click=lambda: st.session_state.clear()):
            st.rerun()

