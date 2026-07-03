"""Live demo: fetch a Wikipedia article as it existed at two different dates.

Shows the time-mask working against the real MediaWiki API: the same title at two
as-of dates returns two different revisions, each guaranteed no newer than its Clock.

Run:  python examples/wikipedia_as_of.py
"""

from forecast_playground import Clock, WikipediaSource


def main() -> None:
    wiki = WikipediaSource()
    title = "ChatGPT"

    for date in ["2023-01-01", "2024-06-01"]:
        clock = Clock.at(date)
        docs = wiki.fetch(title, clock)
        if not docs:
            print(f"as of {date}: article did not exist yet")
            continue
        d = docs[0]
        assert d.timestamp <= clock.as_of, "LEAK: revision newer than as_of!"
        print(f"as of {date}:")
        print(f"  revision {d.meta['revid']} @ {d.timestamp.isoformat()}")
        print(f"  length {len(d.content):,} chars  url {d.url}")
        print()


if __name__ == "__main__":
    main()
