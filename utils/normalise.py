def normalise_url(url: str):
    if not isinstance(url, str): return url
    url = url.strip().lower()
    if url.endswith('/'): url = url[:-1]
    if '?' in url: url = url.split('?')[0]
    return url