"""Grounded Gemini research call — one call per supplier (+ optional conditional retry)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types

from shared.config import Settings, get_settings
from shared.models import FACILITY_LIST_SCHEMA, PRODUCT_LIST_SCHEMA

logger = logging.getLogger("frp.gemini")

# DISCOVER_PRODUCT: a cheap, single grounded call used only when the caller didn't
# supply a product. Deliberately tiny output cap (see discover_products) — this is
# a short ranked name list, not a facility search.
PRODUCT_DISCOVERY_SYSTEM_PROMPT = (
    "You are a supply-chain research assistant. Given a supplier company with no product "
    "specified, identify the products it is most known for producing or most commonly sourced "
    "for — ranked most to least significant by production volume, revenue, or sourcing frequency. "
    "Return a JSON array of up to 5 short product names (e.g. \"cold-rolled steel coil\", "
    "\"organic cotton yarn\"), most significant first. If the company is genuinely unknown, infer "
    "typical products from its industry sector or trade classification rather than returning an "
    "empty list. Return JSON only."
)
PRODUCT_DISCOVERY_MAX_OUTPUT_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a supply-chain research assistant. Given a supplier company and a product, find every "
    "real, currently operating facility involved in producing that product for the supplier — "
    "including facilities the supplier owns directly AND third-party sites operating on its behalf "
    "(contract manufacturers, toll manufacturers, joint-venture plants, third-party logistics "
    "providers, co-packers). Do not include the customer's own facilities, resellers, or pure "
    "sales/admin offices with no production, storage, or processing role. "
    "For each facility return: facility_name, full postal facility_address, latitude/longitude "
    "if a source states them (else null — never guess coordinates), facility_type as exactly one of "
    "manufacturing | logistics | raw_material, confidence (high/medium/low), and source_url. "
    "Manufacturing = production/assembly/fabrication plants, contract or toll manufacturing sites. "
    "Logistics = warehouses, distribution centers, fulfillment hubs, ports, freight terminals. "
    "Raw_material = mines, quarries, wells, smelters, refineries, mills, tanneries, farms, "
    "plantations, or other primary-material extraction/processing sites. "
    "Search broadly and check multiple source types: the company's own factory/locations pages, "
    "annual reports, ESG/CSRD disclosures, investor filings, published supplier or factory lists "
    "(e.g. OpenSupplyHub, Sourcemap), industry directories, and trade/customs records. Find as many "
    "distinct real facilities as the evidence supports — do not stop after just one or two hits if "
    "more exist. If you cannot verify an address, set confidence low. Return JSON only."
)


@dataclass
class GeminiResult:
    facilities_raw: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    search_queries: int = 0
    raw_text: str = ""


@dataclass
class ProductDiscoveryResult:
    products: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    search_queries: int = 0


def _count_search_queries(response: Any) -> int:
    """Extract number of Google Search queries from grounding metadata."""
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return 0
        meta = getattr(candidates[0], "grounding_metadata", None)
        if meta is None:
            return 0
        queries = getattr(meta, "web_search_queries", None) or getattr(
            meta, "retrieval_queries", None
        )
        if queries:
            return len(list(queries))
        chunks = getattr(meta, "grounding_chunks", None)
        if chunks:
            return len(list(chunks))
        return 0
    except Exception:
        logger.exception("failed to parse grounding metadata")
        return 0


def _usage(response: Any) -> tuple[int, int, int]:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return 0, 0, 0
    inp = int(getattr(meta, "prompt_token_count", 0) or 0)
    out = int(getattr(meta, "candidates_token_count", 0) or 0)
    think = int(getattr(meta, "thoughts_token_count", 0) or 0)
    return inp, out, think


def _parse_facilities(text: str) -> list[dict[str, Any]]:
    if not text or not text.strip():
        return []
    data = json.loads(text)
    if isinstance(data, dict) and "facilities" in data:
        data = data["facilities"]
    if not isinstance(data, list):
        return []
    return data[:40]


def _parse_products(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    data = json.loads(text)
    if isinstance(data, dict) and "products" in data:
        data = data["products"]
    if not isinstance(data, list):
        return []
    return [str(p).strip() for p in data if str(p).strip()][:5]


class GeminiClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = genai.Client(api_key=self.settings.gemini_api_key)

    def _build_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            response_mime_type="application/json",
            response_schema=FACILITY_LIST_SCHEMA,
            max_output_tokens=self.settings.max_output_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.settings.thinking_budget
            ),
            temperature=0.1,
        )

    async def research(self, supplier_name: str, product: str) -> GeminiResult:
        user_msg = f"Supplier: {supplier_name}\nProduct: {product}"
        result = await self._call(user_msg)

        # Conditional retry (max 1): low yield on the first call. A different search
        # angle — third-party/investigative sources instead of the company's own
        # pages — tends to surface facilities the first pass missed.
        if len(result.facilities_raw) < self.settings.min_facilities_before_retry:
            logger.info(
                "conditional retry for %s / %s (found=%s, searches=%s)",
                supplier_name,
                product,
                len(result.facilities_raw),
                result.search_queries,
            )
            known = ", ".join(
                f.get("facility_name", "") for f in result.facilities_raw if f.get("facility_name")
            ) or "none yet"
            retry_msg = (
                f"{user_msg}\n"
                f"Facilities already found (do not repeat): {known}\n"
                "The first pass may have missed facilities. Broaden the search: check contract/toll "
                "manufacturer relationships, third-party logistics providers, published factory-audit "
                "databases (OpenSupplyHub, Sourcemap, Better Work, Fair Labor Association), import/"
                f'customs records, and local trade press for "{supplier_name}" facility locations. '
                "Only return facilities not already found."
            )
            retry = await self._call(retry_msg)
            result.facilities_raw = result.facilities_raw + retry.facilities_raw
            result.input_tokens += retry.input_tokens
            result.output_tokens += retry.output_tokens
            result.thinking_tokens += retry.thinking_tokens
            result.search_queries += retry.search_queries
            result.raw_text = retry.raw_text

        return result

    async def discover_products(self, supplier_name: str) -> ProductDiscoveryResult:
        """DISCOVER_PRODUCT: one small grounded call (+ optional conditional retry)
        to find the products a supplier is most known for or sourced for, used only
        when the caller didn't supply a product."""
        user_msg = f"Supplier: {supplier_name}"
        result = await self._call_product_discovery(user_msg)

        if not result.products:
            retry = await self._call_product_discovery(
                f"{user_msg}\nThis company may not be widely known. Infer likely products from its "
                "industry sector, SEC/annual-report filings, or trade classification instead of "
                "returning an empty list."
            )
            result.products = retry.products
            result.input_tokens += retry.input_tokens
            result.output_tokens += retry.output_tokens
            result.thinking_tokens += retry.thinking_tokens
            result.search_queries += retry.search_queries

        return result

    async def _call_product_discovery(self, user_message: str) -> ProductDiscoveryResult:
        config = types.GenerateContentConfig(
            system_instruction=PRODUCT_DISCOVERY_SYSTEM_PROMPT,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            response_mime_type="application/json",
            response_schema=PRODUCT_LIST_SCHEMA,
            max_output_tokens=PRODUCT_DISCOVERY_MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=self.settings.thinking_budget),
            temperature=0.1,
        )
        response = await self._client.aio.models.generate_content(
            model=self.settings.gemini_model,
            contents=user_message,
            config=config,
        )
        text = getattr(response, "text", None) or ""
        inp, out, think = _usage(response)
        searches = _count_search_queries(response)
        try:
            products = _parse_products(text)
        except json.JSONDecodeError:
            logger.warning("Gemini returned non-JSON for product discovery; treating as empty")
            products = []
        return ProductDiscoveryResult(
            products=products,
            input_tokens=inp,
            output_tokens=out,
            thinking_tokens=think,
            search_queries=searches,
        )

    async def _call(self, user_message: str) -> GeminiResult:
        config = self._build_config()
        response = await self._client.aio.models.generate_content(
            model=self.settings.gemini_model,
            contents=user_message,
            config=config,
        )
        text = getattr(response, "text", None) or ""
        inp, out, think = _usage(response)
        searches = _count_search_queries(response)
        try:
            facilities = _parse_facilities(text)
        except json.JSONDecodeError:
            logger.warning("Gemini returned non-JSON; treating as empty")
            facilities = []
        return GeminiResult(
            facilities_raw=facilities,
            input_tokens=inp,
            output_tokens=out,
            thinking_tokens=think,
            search_queries=searches,
            raw_text=text,
        )
