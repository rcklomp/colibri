# x_read.py -- read-only X (twitter-cli) wrapper: passes the Opera session cookies (auth_token, ct0 only) through the child environment, never prints or writes them.
import os, subprocess, sys
import browser_cookie3
ALLOWED = {"search", "tweet", "user", "user-posts", "article"}          # read commands only; no post/like/follow/retweet/delete/feed/bookmarks
if not sys.argv[1:] or sys.argv[1] not in ALLOWED:
    sys.exit("read-only commands only: %s" % sorted(ALLOWED))
CF = os.path.expanduser("~/Library/Application Support/com.operasoftware.Opera/Default/Cookies")
want = {}
for dom in ("x.com", "twitter.com"):
    for c in browser_cookie3.opera(cookie_file=CF, domain_name=dom):
        if c.name in ("auth_token", "ct0") and c.value: want.setdefault(c.name, c.value)
env = dict(os.environ, TWITTER_AUTH_TOKEN=want["auth_token"], TWITTER_CT0=want["ct0"], TWITTER_BROWSER="")
r = subprocess.run([os.path.expanduser("~/.agent-reach/tools/twitter-venv/bin/twitter"), "-c"] + sys.argv[1:], env=env, capture_output=True, text=True, timeout=120)
out = (r.stdout or "") + (r.stderr or "")
for v in want.values(): out = out.replace(v, "<redacted>")
print(out[:9000])
