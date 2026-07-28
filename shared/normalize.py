from slugify import slugify


def make_normalized_key(supplier_name: str, product: str) -> str:
    """Canonical *per-run* dedupe key for a (supplier, product) pair.

    Used for Redis enqueue dedupe and the Supabase per-product cache lookup —
    "never research the same (supplier, product) twice inside the refresh window".
    """
    return f"{make_normalized_name(supplier_name)}:{slugify(product.lower())}"


def make_normalized_name(supplier_name: str) -> str:
    """Canonical *supplier identity* key — one company, independent of product.

    A supplier can supply many products, so its Supabase row is keyed on this, not
    on (name, product). The SQL backfill in migration 003 mirrors this slug for
    ASCII names; unicode/edge cases may differ and simply cause a one-time cache miss.
    """
    return slugify(supplier_name.lower())


def normalize_address(address: str) -> str:
    """Normalize an address string for geocode cache keys."""
    return " ".join(address.lower().split())
