import sqlite3
import pylast
from flask import Flask, redirect, request, session, url_for, jsonify

app = Flask(__name__)
app.secret_key = 'replace_with_your_secret_key'

API_KEY = "your_lastfm_api_key"
API_SECRET = "your_lastfm_api_secret"

DB_PATH = "yourdatabase.db"

def create_lastfm_sessions_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS LASTFM_SESSIONS (
            userid TEXT PRIMARY KEY,
            lastfm_username TEXT NOT NULL,
            session_key TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_lastfm_session(userid, lastfm_username, session_key):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO LASTFM_SESSIONS (userid, lastfm_username, session_key)
        VALUES (?, ?, ?)
    """, (userid, lastfm_username, session_key))
    conn.commit()
    conn.close()

def get_lastfm_session(userid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT lastfm_username, session_key FROM LASTFM_SESSIONS WHERE userid=?
    """, (userid,))
    row = c.fetchone()
    conn.close()
    if row:
        return {'lastfm_username': row[0], 'session_key': row[1]}
    return None

def get_lastfm_network(session_key):
    return pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET, session_key=session_key)

create_lastfm_sessions_table()

@app.route('/lastfm/login')
def index():
    # For demo: simulate internal user id stored in session (replace with real login)
    if 'internal_userid' not in session:
        session['internal_userid'] = 'example_userid_123'

    userid = session['internal_userid']
    existing = get_lastfm_session(userid)
    if existing:
        return (
            f"Last.fm connected as {existing['lastfm_username']}. "
            f"<a href='/lastfm/recent'>Recent Tracks</a> | "
            f"<a href='/lastfm/loved'>Loved Tracks</a> | "
            f"<a href='/lastfm/top-artists'>Top Artists</a>"
        )

    network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET)
    token = network.get_authentication_token()
    session['lastfm_token'] = token
    auth_url = pylast.auth_url(token)
    return redirect(auth_url)

@app.route('/lastfm/callback')
def callback():
    token = session.get('lastfm_token')
    userid = session.get('internal_userid')
    if not token or not userid:
        return "Missing token or user session.", 400

    network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET)
    session_key = network.get_session_key(token)
    if not session_key:
        return "Failed to get session key.", 400

    authed_network = get_lastfm_network(session_key)
    user = authed_network.get_authenticated_user()
    lastfm_username = user.get_name()

    save_lastfm_session(userid, lastfm_username, session_key)

    return (
        f"Successfully connected Last.fm account {lastfm_username}! "
        f"<a href='/lastfm/recent'>See recent tracks</a>"
    )

def handle_api_call(userid, fetch_func):
    session_data = get_lastfm_session(userid)
    if not session_data:
        return redirect(url_for('index'))  # Not connected, start auth

    network = get_lastfm_network(session_data['session_key'])
    user = network.get_authenticated_user()

    try:
        data = fetch_func(user)
    except pylast.WSError:
        # Probably revoked or invalid session
        # Remove session and prompt re-auth
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM LASTFM_SESSIONS WHERE userid=?", (userid,))
        conn.commit()
        conn.close()
        return {
            'error': 'Last.fm session invalid or revoked. Please reconnect.',
            're_auth_url': url_for('index', _external=True)
        }, 403

    return data

@app.route('/lastfm/recent')
def lastfm_recent():
    userid = session.get('internal_userid')
    if not userid:
        return "Not logged in.", 401

    def fetch_recent(user):
        recent_tracks = user.get_recent_tracks(limit=5)
        tracks_list = []
        for track in recent_tracks:
            tracks_list.append({
                'artist': track.track.artist.name,
                'title': track.track.title,
                'album': track.album,
                'played_at': track.timestamp  # may be None if currently playing
            })
        return {'lastfm_username': user.get_name(), 'recent_tracks': tracks_list}

    return jsonify(handle_api_call(userid, fetch_recent))

@app.route('/lastfm/loved')
def lastfm_loved():
    userid = session.get('internal_userid')
    if not userid:
        return "Not logged in.", 401

    def fetch_loved(user):
        loved_tracks = user.get_loved_tracks()
        tracks_list = []
        for track in loved_tracks:
            tracks_list.append({
                'artist': track.track.artist.name,
                'title': track.track.title,
                'album': track.album,
                'date_loved': track.date
            })
        return {'lastfm_username': user.get_name(), 'loved_tracks': tracks_list}

    return jsonify(handle_api_call(userid, fetch_loved))

@app.route('/lastfm/top-artists')
def lastfm_top_artists():
    userid = session.get('internal_userid')
    if not userid:
        return "Not logged in.", 401

    def fetch_top_artists(user):
        top_artists = user.get_top_artists(limit=5)
        artists_list = []
        for artist, playcount in top_artists:
            artists_list.append({
                'name': artist.name,
                'playcount': playcount
            })
        return {'lastfm_username': user.get_name(), 'top_artists': artists_list}

    return jsonify(handle_api_call(userid, fetch_top_artists))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=444555, debug=True)
