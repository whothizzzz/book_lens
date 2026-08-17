"""
Vercel Python Entrypoint for BookLens
Exposes a WSGI handler `app` for Vercel Functions.
"""

from __future__ import annotations

import json
from urllib.parse import parse_qs


def app(environ: dict, start_response) -> list[bytes]:
    """WSGI application entrypoint for Vercel Serverless Functions."""
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET").upper()
    query_string = environ.get("QUERY_STRING", "")
    params = parse_qs(query_string)

    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type"),
    ]

    if method == "OPTIONS":
        start_response("204 No Content", headers)
        return []

    # Health check route
    if path in ("/api/health", "/health"):
        start_response("200 OK", headers)
        response_body = {"status": "ok", "service": "BookLens API"}
        return [json.dumps(response_body).encode("utf-8")]

    # Live book search route
    if path in ("/api/search", "/search"):
        query = params.get("q", [""])[0] or params.get("query", [""])[0]
        max_results = int(params.get("max_results", [10])[0])

        if not query:
            start_response("400 Bad Request", headers)
            response_body = {"error": "Query parameter 'q' is required"}
            return [json.dumps(response_body).encode("utf-8")]

        try:
            from src.book_api import fetch_live_books

            books = fetch_live_books(query=query, max_results=min(max_results, 20))
            start_response("200 OK", headers)
            response_body = {
                "query": query,
                "count": len(books),
                "results": books,
            }
            return [json.dumps(response_body).encode("utf-8")]
        except Exception as exc:
            start_response("500 Internal Server Error", headers)
            response_body = {"error": f"Search failed: {str(exc)}"}
            return [json.dumps(response_body).encode("utf-8")]

    # Root / Info route
    start_response("200 OK", headers)
    response_body = {
        "name": "BookLens API",
        "description": "Semantic Multilingual Book Recommendation Engine",
        "endpoints": {
            "/api/health": "Health check",
            "/api/search?q=<query>": "Search live books via Google Books & Open Library",
        },
    }
    return [json.dumps(response_body, indent=2).encode("utf-8")]
