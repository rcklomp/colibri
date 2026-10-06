#!/usr/bin/env python3
"""reddit_rss.py -- read-only Reddit research over the PUBLIC RSS/Atom feeds (python3 stdlib; no login, no cookies, no account).

    reddit_rss.py search SUBREDDIT "query" [limit=8]     relevance-sorted posts of a subreddit matching the query
    reddit_rss.py thread URL_OR_ID [max_comments=25]    one post and its comments (the thread's own .rss)
    reddit_rss.py batch FILE                             lines "SUBREDDIT<TAB>query"; paced

Why: reddit.com is closed to WebFetch / WebSearch and to Jina's reader (403 "network policy"), but the public feeds answer a plain request
with an honest User-Agent; they rate-limit hard (HTTP 429 after a few requests in a few seconds), so requests are spaced (MIN_GAP seconds),
Retry-After is honoured and a 429 backs off instead of hammering. The User-Agent names the tool only: no person, no address.
"""
import re, sys, time, html, urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET

UA = "claude-code-research/1.0 (read-only technical research, public RSS feeds)"
NS = {"a": "http://www.w3.org/2005/Atom"}
MIN_GAP = 18.0
_last = [0.0]


def get(url, tries=4):
    for k in range(tries):
        wait = _last[0] + MIN_GAP - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=40) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                back = float(ra) if ra and ra.replace(".", "").isdigit() else 60.0 * (k + 1)
                print("  [429, waiting %.0f s]" % back, file=sys.stderr)
                time.sleep(back)
                continue
            print("  [HTTP %d for %s]" % (e.code, url[:90]), file=sys.stderr)
            return None
        except Exception as e:
            print("  [error %s]" % str(e)[:80], file=sys.stderr)
            return None
    return None


def text_of(h, n):
    t = re.sub(r"<[^>]+>", " ", html.unescape(h or ""))
    return re.sub(r"\s+", " ", html.unescape(t)).strip()[:n]


def entries(xml):
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    out = []
    for e in root.findall("a:entry", NS):
        link = e.find("a:link", NS)
        out.append({
            "title": (e.findtext("a:title", "", NS) or "").strip(),
            "href": link.get("href") if link is not None else "",
            "updated": (e.findtext("a:updated", "", NS) or "")[:10],
            "author": (e.findtext("a:author/a:name", "", NS) or "").strip(),
            "content": e.findtext("a:content", "", NS) or "",
        })
    return out


def search(sub, q, limit=8, snippet=240):
    url = "https://www.reddit.com/r/%s/search.rss?q=%s&restrict_sr=on&sort=relevance&t=all" % (sub, urllib.parse.quote_plus(q))
    xml = get(url)
    print("### r/%s : %s" % (sub, q))
    if xml is None:
        print("  (no answer)"); return
    es = entries(xml)[:limit]
    for e in es:
        print("  [%s] %s\n      %s\n      %s" % (e["updated"], e["title"][:130], e["href"], text_of(e["content"], snippet)))
    if not es:
        print("  (no results)")


def thread(ref, max_comments=25, body=900):
    m = re.search(r"comments/([a-z0-9]+)", ref)
    pid = m.group(1) if m else ref
    xml = get("https://www.reddit.com/comments/%s/.rss?limit=%d" % (pid, max_comments))
    if xml is None:
        print("(no answer)"); return
    es = entries(xml)
    for i, e in enumerate(es):
        print("%s[%s] %s  by %s\n%s\n" % ("POST " if i == 0 else "  reply ", e["updated"], e["title"][:120] if i == 0 else "", e["author"],
                                          text_of(e["content"], body if i == 0 else 600)))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "search":
        search(a[1], a[2], int(a[3]) if len(a) > 3 else 8)
    elif len(a) >= 2 and a[0] == "thread":
        thread(a[1], int(a[2]) if len(a) > 2 else 25)
    elif len(a) >= 2 and a[0] == "batch":
        for ln in open(a[1]):
            if ln.strip() and not ln.startswith("#"):
                sub, q = ln.rstrip("\n").split("\t", 1); search(sub, q); sys.stdout.flush()
    else:
        sys.exit(__doc__)
