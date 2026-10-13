"""Deterministic evidence verifier used at the planner termination boundary."""
import json
import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

from .models import VerificationResult
from .provenance import time_ranges, observation_ranges, ranges_cover, citation_ranges
from core.answer_contract import normalize_answer_format


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "evolution.yaml"

_ANSWER_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "to", "of", "and", "or", "option", "answer",
    "at", "for", "from", "her", "his", "their", "its", "our", "your", "my",
    "i", "we", "you", "he", "she", "they",
    "in", "it", "on", "that", "this", "where", "while", "with",
})
_CONTENT_TOKEN_EQUIVALENTS = {
    "clean": frozenset({"wash", "rinse"}),
    "wash": frozenset({"clean", "rinse"}),
    "rinse": frozenset({"clean", "wash"}),
    "cut": frozenset({"slice", "trim"}),
    "slice": frozenset({"cut", "trim"}),
    "trim": frozenset({"cut", "slice"}),
    "foot": frozenset({"feet"}),
    "feet": frozenset({"foot"}),
    "even": frozenset({"evenly"}),
    "evenly": frozenset({"even"}),
}
_DYNAMIC_GENERIC_TOKENS = frozenset({
    "action", "appear", "caption", "computer", "frame", "hand", "laptop",
    "screen", "segment", "transcript", "video", "visual", "woman",
})
# A dynamic QA backend may answer ``hands a gift`` while the benchmark option
# says ``delivers a product``.  Literal token overlap is too strict for this
# safe, action-specific paraphrase, but a broad semantic matcher would make a
# wrong option look supported.  The verifier therefore requires one matching
# transfer-action group *and* one matching object group.
_DYNAMIC_TRANSFER_ACTION_TOKENS = frozenset({
    "deliver", "give", "hand", "offer", "pass", "present", "send", "transfer",
})
_DYNAMIC_TRANSFER_OBJECT_TOKENS = frozenset({
    "gift", "item", "merchandise", "package", "present", "product", "box",
})
_OPTION_LINE_RE = re.compile(r"^\s*\(([A-Z])\)\s+(.+?)\s*$", re.MULTILINE)
_COMPARISON_MARKERS = frozenset({
    "against", "alternative", "although", "but", "compare", "compared", "conflict",
    "comparison", "contradict", "different", "difference", "instead", "not", "other",
    "others", "rather", "versus", "vs", "whereas", "while",
})
_NEGATION_MARKERS = frozenset({
    "absent", "lack", "lacks", "missing", "no", "none", "not", "without",
})
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}
_COUNT_QUERY_MARKERS = frozenset({
    "how many", "number of", "count", "counting problem", "action count", "moving count",
})
_COMPARATIVE_COUNT_RE = re.compile(
    r"\b(?:largest|smallest|highest|lowest|greatest|fewest|most|least)\s+"
    r"(?:number|quantity|count|amount)\b",
    re.IGNORECASE,
)
_COUNT_COMPARISON_MARKERS = frozenset({
    "largest", "smallest", "highest", "lowest", "greatest", "fewest", "most",
    "least", "more", "fewer", "greater", "greater-than", "maximum", "minimum",
    "max", "min", "quantity", "amount",
})
_PURPOSE_QUERY_MARKERS = (
    "purpose", "function", "used for", "used to", "role of", "why was", "why did",
)
_RATIONALE_QUERY_MARKERS = (
    "importance", "important", "significance", "significant", "why", "because",
    "demonstrated", "demonstrate", "infer", "inferred", "inference", "ensure",
    "ensures", "goal", "objective", "reason",
)
_NEGATIVE_QUERY_MARKERS = (
    " not ", "does not", "doesn't", "did not", "didn't", "never", "without",
    "absent", "which one was not", "what did .* not", "none of the above",
)
_UNCERTAINTY_MARKERS = frozenset({
    "approximate", "approx", "uncertain", "unclear", "unknown", "insufficient",
    "ambiguous", "maybe", "possibly", "not sure", "low", "medium",
    "low confidence", "medium confidence",
})
_NATIVE_COUNT_SOURCE_TOOLS = frozenset({
    "Video_Clip_QA", "Aggregation", "Clean_Results",
})
_SYNTHETIC_ONLY_SOURCE_TOOLS = frozenset({"Python_Executor", "SQL_Executor"})
_SOURCE_TOOL_CANONICAL = {
    name.casefold(): name
    for name in {
        "Inspect_Frame", "Inspect_Clip", "Crop", "Zoom_in", "Video_Clip_QA",
        "Video_Summary", "Action_Recognition", "Cross_Modal_Verify",
        "Semantic_Retrieve", "Keyword_Retrieve", "Segment_Retrieve", "Graph_Search",
        "Python_Executor", "SQL_Executor",
    }
}


def _canonical_source_tool(value: Any) -> str:
    """Normalize executable source names while preserving unknown names."""
    text = str(value or "").strip()
    return _SOURCE_TOOL_CANONICAL.get(text.casefold(), text)
_ACTION_QUERY_MARKERS = frozenset({
    "put", "puts", "place", "placed", "add", "adds", "added", "remove", "removes",
    "removed", "insert", "inserts", "inserted", "pour", "pours", "poured", "take", "takes",
    "took", "pick", "picks", "picked",
})
_TOTAL_CONTEXT_MARKERS = frozenset({
    "already", "existing", "exists", "contain", "contains", "contained", "total",
    "altogether", "resulting", "afterward", "afterwards", "visible", "present",
})
_DIRECT_ACTION_COUNT_RE = re.compile(
    r"\b(?:put|puts|place|places|placed|add|adds|added|remove|removes|removed|insert|inserts|inserted|"
    r"pour|pours|poured|take|takes|took|pick|picks|picked)\b"
    r"\s+(?:exactly\s+)?(?:(?:a|an)\s+)?(?:single|one|two|three|four|five|"
    r"six|seven|eight|nine|ten|\d+)\b",
    re.IGNORECASE,
)
def _has_dynamic_relation(text: str) -> bool:
    """Recognize action predicates without treating body parts/directions as verbs."""
    normalized = re.sub(r"\b(?:on|to|at)\s+(?:the\s+)?left\b|\bleft[- ](?:hand|side)\b", "", text.lower())
    return bool(re.search(
        r"\b(?:carry|carries|carried|carrying|take|takes|taking|took|taken|"
        r"bring|brings|bringing|brought|give|gives|giving|gave|given|handed|handing|"
        r"pass|passes|passed|passing|transfer(?:s|red|ring)?|rescue[ds]?|rescuing|"
        r"help(?:s|ed|ing)?|leave|leaves|leaving|escape[ds]?|escaping|move[ds]?|moving|"
        r"jump(?:s|ed|ing)?|share[sd]?|hug(?:s|ged|ging)?|bow(?:s|ed|ing)?|"
        r"clap(?:s|ped|ping)?|wave[sd]?|smile[sd]?|laugh(?:s|ed|ing)?|dance[sd]?|"
        r"eat(?:s|ing)?|ate|drink(?:s|ing)?|drank|open(?:s|ed|ing)?|close[sd]?|"
        r"put(?:s|ting)?|pick(?:s|ed|ing)?|throw(?:s|ing)?|threw|catch(?:es|ing)?|caught|"
        r"push(?:es|ed|ing)?|pull(?:s|ed|ing)?|touch(?:es|ed|ing)?|hold(?:s|ing)?|held|"
        r"wear(?:s|ing)?|wore|look(?:s|ed|ing)?|watch(?:es|ed|ing)?|talk(?:s|ed|ing)?|"
        r"speak(?:s|ing)?|spoke|sing(?:s|ing)?|sang|play(?:s|ed|ing)?|write[sn]?|wrote|"
        r"read(?:s|ing)?|cut(?:s|ting)?|cross(?:es|ed|ing)?|enter(?:s|ed|ing)?|exit(?:s|ed|ing)?|"
        r"sit(?:s|ting)?|sat|stand(?:s|ing)?|stood|walk(?:s|ed|ing)?|run(?:s|ning)?|ran|"
        r"turn(?:s|ed|ing)?|raise[sd]?|point(?:s|ed|ing)?|kiss(?:es|ed|ing)?|embrace[sd]?|"
        r"before and after|during the transition)\b|"
        r"\b(?:he|she|they|person|man|woman|protagonist)\s+left\b|"
        r"\bwhat\s+(?:does|did|do)\b[^?!.]*\bdo\b|"
        r"\bwhat\s+(?:is|are|was|were)\b[^?!.]*\bdoing\b|"
        r"\b(?:what|which)\s+actions?\b|\bwhat\s+happens?\b|"
        r"\bhands?\s+(?:a|an|the)\b[^?!.]{0,60}\b(?:to|over)\b",
        normalized,
    ))


def _dynamic_transfer_paraphrase_match(
    option_tokens: set[str],
    evidence_tokens: set[str],
) -> bool:
    """Match only the narrow transfer-action/product paraphrase contract."""

    return bool(
        option_tokens.intersection(_DYNAMIC_TRANSFER_ACTION_TOKENS)
        and evidence_tokens.intersection(_DYNAMIC_TRANSFER_ACTION_TOKENS)
        and option_tokens.intersection(_DYNAMIC_TRANSFER_OBJECT_TOKENS)
        and evidence_tokens.intersection(_DYNAMIC_TRANSFER_OBJECT_TOKENS)
    )


def _slot_source_tools(item: Mapping[str, Any]) -> set[str]:
    values: List[Any] = [item.get("source_tool")]
    raw = item.get("supporting_source_tools")
    if isinstance(raw, (list, tuple, set)):
        values.extend(raw)
    elif raw:
        values.append(raw)
    return {
        _canonical_source_tool(value)
        for value in values
        if str(value or "").strip()
    }


def _slot_is_synthetic_only(item: Mapping[str, Any]) -> bool:
    """Return true when a claim is sourced solely from executable narration."""
    sources = _slot_source_tools(item)
    return bool(sources) and sources.issubset(_SYNTHETIC_ONLY_SOURCE_TOOLS)


def _slot_has_synthetic_provenance(item: Mapping[str, Any]) -> bool:
    """Reject claims whose provenance is contaminated by executable narration.

    An Observer may copy a real tool name into ``source_tool`` while citing a
    later ``Python_Executor`` interpretation.  Treating that mixed record as
    direct visual evidence lets derived text manufacture a purpose/count claim
    that the native tool never made.  Any explicit synthetic source in the
    source list or citation therefore invalidates the record for grounding.
    """
    sources = _slot_source_tools(item)
    if sources.intersection(_SYNTHETIC_ONLY_SOURCE_TOOLS):
        return True
    citation_blob = " ".join(
        str(item.get(key) or "") for key in ("citation", "supporting_citations")
    )
    return bool(
        re.search(r"\b(?:Python_Executor|SQL_Executor)\b", citation_blob, re.IGNORECASE)
    )


def _cited_image_assets(citation: str) -> set[str]:
    """Extract explicit image identities, never generic frame/pack labels."""
    return {
        Path(match).name
        for match in re.findall(
            r"[\w./-]+\.(?:jpe?g|png|webp|bmp)(?![\w.])",
            citation, re.IGNORECASE,
        )
    }


def _observation_image_assets(observation: Mapping[str, Any]) -> set[str]:
    payload = _coerce_mapping(observation.get("result"))
    assets: set[str] = set()
    for record in (observation, payload):
        images = record.get("image_evidence") or []
        if not isinstance(images, list):
            continue
        for image in images:
            if isinstance(image, Mapping):
                path = image.get("image") or image.get("image_path")
                if path:
                    assets.add(Path(str(path)).name)
    return assets


def _observation_source_ref(observation: Mapping[str, Any]) -> str:
    """Stable source identity across timing/debug metadata and record wrappers."""
    identity = {'tool': _canonical_source_tool(observation.get('tool')),
                'assets': sorted(_observation_image_assets(observation)),
                'ranges': sorted(set(observation_ranges(observation)))}
    # Native results are rendered as text for Planner and dictionaries in some
    # tool records. Bind the physical source, not that presentation difference;
    # rereading the same source must not manufacture an independent observation.
    return 'observation:' + hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


def _slot_has_executed_source(item: Mapping[str, Any], steps: Iterable[Any]) -> bool:
    """Bind a cited claim to an explicitly successful tool observation.

    Legacy citations may name a tool instead of declaring source_tool, but
    a bare timestamp or a self-reported verified flag is never provenance.
    """
    if _slot_is_synthetic_only(item) or _slot_has_synthetic_provenance(item):
        return False
    citation = " ".join(
        str(item.get(key) or "") for key in ("citation", "supporting_citations")
    ).strip()
    if not citation:
        return False
    declared = _slot_source_tools(item)
    cited_assets = _cited_image_assets(citation)
    matched_assets: set[str] = set()
    matched_ranges = []
    window = item.get("time_window")
    claimed_ranges = time_ranges(window)
    if isinstance(window, (Mapping, list, tuple)) and window and not claimed_ranges:
        return False
    claimed_ranges += citation_ranges(item.get("citation"))
    matched_source = False
    requested_refs = item.get('source_observation_refs') or []
    if not isinstance(requested_refs, list) or any(not isinstance(ref, str) for ref in requested_refs):
        return False
    seen_refs = set()
    for step in steps:
        observations = (
            step.get("observation_items") if isinstance(step, Mapping)
            else getattr(step, "observation_items", None)
        ) or []
        for observation in _iter_observation_items(observations):
            tool = _canonical_source_tool(observation.get("tool"))
            if not tool or str(observation.get("status") or "").strip().lower() != "success":
                continue
            source_ref = _observation_source_ref(observation)
            if requested_refs and source_ref not in requested_refs:
                continue
            seen_refs.add(source_ref)
            assets = _observation_image_assets(observation)
            if cited_assets and not cited_assets.intersection(assets):
                # A different call of the same tool cannot lend its time window
                # to the specifically cited image from another call.
                continue
            if declared:
                if tool in declared:
                    matched_source = True
                    matched_assets.update(assets)
                    matched_ranges.extend(observation_ranges(observation))
            elif re.search(rf"(?<!\w){re.escape(tool)}(?!\w)", citation, re.IGNORECASE):
                matched_source = True
                matched_assets.update(assets)
                matched_ranges.extend(observation_ranges(observation))
            elif not declared:
                # Legacy frame citations may name the returned asset rather
                # than its tool. Match an actual asset, never a generic pack
                # prefix or an arbitrary timestamp.
                payload = _coerce_mapping(observation.get("result"))
                images = observation.get("image_evidence") or payload.get("image_evidence") or []
                if not isinstance(images, list):
                    continue
                for image in images:
                    if not isinstance(image, Mapping):
                        continue
                    path = str(image.get("image") or image.get("image_path") or "")
                    name = Path(path).name if path else ""
                    if name and re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w.-])", citation):
                        matched_source = True
                        matched_assets.update(assets)
                        matched_ranges.extend(observation_ranges(observation))
    # A real tool invocation cannot authenticate a fabricated image. Multiple
    # cited images may legitimately come from separate successful calls.
    return (matched_source
            and set(requested_refs).issubset(seen_refs)
            and (not cited_assets or cited_assets.issubset(matched_assets))
            and ranges_cover(claimed_ranges, matched_ranges))


def _slot_is_uncertain(item: Mapping[str, Any]) -> bool:
    uncertainty = str(item.get("uncertainty") or "").strip().lower()
    return any(marker in uncertainty for marker in _UNCERTAINTY_MARKERS)


def _slot_has_direct_visual_source(item: Mapping[str, Any], steps=None) -> bool:
    """Whether a ledger claim cites a direct visual/native observation."""
    direct_tools = {
        "Inspect_Frame", "Inspect_Clip", "Crop", "Zoom_in", "Video_Clip_QA",
        "Action_Recognition", "Cross_Modal_Verify",
    }
    if steps is not None:
        steps = list(steps)
        if not _slot_has_executed_source(item, steps):
            return False
        references = item.get('source_observation_refs') or []
        visual_observations = []
        for step in steps:
            observations = (step.get('observation_items') if isinstance(step, Mapping)
                            else getattr(step, 'observation_items', None)) or []
            for observation in _iter_observation_items(observations):
                if references and _observation_source_ref(observation) not in references:
                    continue
                if (_canonical_source_tool(observation.get('tool')) in direct_tools
                        or _observation_image_assets(observation)):
                    visual_observations.append(observation)
        return _slot_has_executed_source({**item, 'source_observation_refs': []},
                                        [{'observation_items': visual_observations}])
    if _slot_source_tools(item).intersection(direct_tools):
        return True
    citation = " ".join(
        str(item.get(key) or "").lower()
        for key in ("citation", "supporting_citations")
    )
    return any(
        marker in citation
        for marker in ("inspect_frame", "inspect frame", "video_clip_qa", "video clip qa", "visual evidence", "frame evidence")
    )


def _direct_action_relation_support(*, query: str, claim: str) -> bool:
    """Require the evidence to describe the requested action relation.

    Words such as ``placed`` and ``burner`` can occur in separate clauses
    (for example, "placed it in front of his face; a burner is visible").
    That is not evidence that the object was placed *in the burner*. The
    relation must occur in one local phrase.
    """
    normalized_query = _semantic_query_text(query)
    normalized_claim = " ".join(str(claim or "").lower().split())
    action_words = r"put|puts|place|places|placed|add|adds|added|remove|removes|removed|insert|inserts|inserted|pour|pours|poured|take|takes|took|pick|picks|picked"
    query_target_tokens = _content_tokens(normalized_query) - {
        "how", "many", "number", "count", "does", "do", "did", "the",
        "a", "an", "protagonist", "person", "someone", "object", "item",
        "stick", "sticks",
    }
    if not query_target_tokens:
        return bool(_DIRECT_ACTION_COUNT_RE.search(normalized_claim))
    for target in sorted(query_target_tokens, key=len, reverse=True):
        pattern = rf"\b(?:{action_words})\b[^.!?]{{0,45}}\b(?:into|inside|in|onto|on|from)\b\s+(?:the|a|an)?\s*\b{re.escape(target)}\b"
        if re.search(pattern, normalized_claim, flags=re.IGNORECASE):
            return True
    return False


def load_evolution_config(path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = (path or DEFAULT_CONFIG_PATH).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Evolution config must be a mapping: {config_path}")
    return payload


def _successful_families(steps: Iterable[Any]) -> set[str]:
    families: set[str] = set()
    for step in steps:
        if isinstance(step, dict):
            action = str(step.get("action") or "")
            observation = str(step.get("observation") or "").strip().lower()
        else:
            action = str(getattr(step, "action", "") or "")
            observation = str(getattr(step, "observation", "") or "").strip().lower()
        if not action or action in {"Finish", "System", "Parallel"} or observation.startswith("error"):
            continue
        if action in {"Semantic_Retrieve", "Keyword_Retrieve", "Segment_Retrieve", "Video_Summary", "asr"}:
            families.add("text")
        elif action in {"Inspect_Frame", "Inspect_Clip", "Video_Clip_QA", "OCR_Extract", "Action_Recognition"}:
            families.add("visual")
        elif action in {"Cross_Modal_Verify", "evidence_verify"}:
            families.add("cross_modal")
    return families


def _stem_token(token: str) -> str:
    # Normalize common English inflections conservatively. In particular,
    # ``sleeves`` must match a direct observation that says ``sleeve``;
    # stripping ``es`` first would produce the non-word ``sleev`` and make a
    # decisive tool result look unrelated to the selected answer.
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ves") and len(token) > 4:
        return token[:-1]
    if token.endswith("es") and len(token) > 4:
        base = token[:-2]
        if base.endswith(("s", "x", "z", "ch", "sh")):
            return base
        return token[:-1]
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[:-len(suffix)]
    return token


def _content_tokens(value: Any) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
    return {
        _NUMBER_WORDS.get(_stem_token(token), _stem_token(token))
        for token in tokens
        if (len(token) > 1 or token.isdigit()) and token not in _ANSWER_STOPWORDS
    }


def _has_direct_answer_support(
    answer: str,
    verified_slots: Iterable[Mapping[str, Any]],
    query: str = "",
    steps: Iterable[Any] = (),
) -> bool:
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return False
    answer_label_match = re.search(r"\(([A-Z])\)", str(answer or "").upper())
    answer_label = answer_label_match.group(1) if answer_label_match else ""
    options = {
        label: text
        for label, text in _OPTION_LINE_RE.findall(str(query or ""))
    }
    selected_option_tokens = _content_tokens(options.get(answer_label, answer))
    generic_tokens = _DYNAMIC_GENERIC_TOKENS.union({
        "person", "people", "thing", "object", "item", "one", "same", "different",
        "answer", "option", "choice", "yes", "no", "none", "visible", "shown",
    })

    def _overlap_is_decisive(evidence_tokens: set[str]) -> bool:
        overlap = selected_option_tokens.intersection(evidence_tokens)
        for token in selected_option_tokens:
            if _CONTENT_TOKEN_EQUIVALENTS.get(token, frozenset()).intersection(evidence_tokens):
                overlap.add(token)
        specific = overlap - generic_tokens
        # A one-token categorical answer (``Blue``, ``Apples``) is decisive
        # when that token is actually cited.  Longer options need two content
        # tokens unless a native tool returned the complete option text; this
        # prevents a generic word such as ``hand`` or ``visible`` from making
        # a hallucinated multi-clause option appear grounded.
        if len(selected_option_tokens) <= 1:
            return bool(specific or overlap)
        if len(selected_option_tokens) == 2:
            # A single shared noun is usually the common object of several
            # alternatives (for example clearing brush vs pruning the fence).
            # Require both semantic tokens unless the narrow transfer
            # paraphrase rule below establishes the complete relation.
            return len(specific) >= 2
        return (
            len(specific) >= 2
            or selected_option_tokens.issubset(evidence_tokens)
            or (
                len(specific) >= 2
                and any(len(token) >= 6 for token in specific)
            )
        )

    for item in verified_slots:
        if _slot_is_synthetic_only(item):
            continue
        evidence_tokens = _content_tokens(f"{item.get('label', '')} {item.get('value', '')}")
        if _overlap_is_decisive(evidence_tokens) or _dynamic_transfer_paraphrase_match(
            selected_option_tokens,
            evidence_tokens,
        ):
            return True
    # Native tool output is an independent evidence source.  Requiring every
    # useful relation to be copied verbatim into a Planner Evidence slot made
    # the verifier boundary depend on LLM formatting rather than observation
    # content.  Keep the source restricted to successful executable tools.
    for step in steps:
        if isinstance(step, Mapping):
            observation_items = step.get("observation_items") or []
        else:
            observation_items = getattr(step, "observation_items", None) or []
        for item in _iter_observation_items(observation_items):
            if _canonical_source_tool(item.get("tool")) not in {
                "Video_Clip_QA", "Action_Recognition", "Cross_Modal_Verify"
            }:
                continue
            if str(item.get("status") or "").lower() != "success":
                continue
            result_tokens = _content_tokens(item.get("result"))
            if _overlap_is_decisive(result_tokens) or _dynamic_transfer_paraphrase_match(
                selected_option_tokens,
                result_tokens,
            ):
                return True
    return False


def _has_semantic_answer_redundancy(
    *,
    answer_label: str,
    options: Mapping[str, str],
    query: str,
    verified_slots: Iterable[Mapping[str, Any]],
    steps: Iterable[Any],
) -> bool:
    """Require two distinct direct observations for high-risk categorical answers."""
    if not options or not answer_label or answer_label not in options:
        return False
    selected_tokens = _content_tokens(options.get(answer_label, ""))
    if not selected_tokens:
        return False
    generic_tokens = _DYNAMIC_GENERIC_TOKENS.union({
        "person", "people", "thing", "object", "item", "one", "same",
        "different", "answer", "option", "choice", "yes", "no", "none",
        "visible", "shown",
    })

    def _supports(evidence: Any) -> bool:
        evidence_tokens = _content_tokens(evidence)
        overlap = selected_tokens.intersection(evidence_tokens)
        specific = overlap - generic_tokens
        if len(selected_tokens) <= 1:
            return bool(overlap)
        if len(selected_tokens) == 2:
            return len(specific) >= 2
        return (
            selected_tokens.issubset(evidence_tokens)
            or len(specific) >= 2
        )

    step_list = list(steps)
    records: set[tuple[str, str, str, str, str]] = set()
    direct_tools = {
        "Inspect_Frame", "Inspect_Clip", "Crop", "Zoom_in",
        "Video_Clip_QA", "Action_Recognition", "Cross_Modal_Verify",
    }
    for item in verified_slots:
        if not isinstance(item, Mapping):
            continue
        if (
            bool(item.get("context_only", False))
            or bool(item.get("superseded", False))
            or _slot_is_synthetic_only(item)
            or _slot_is_uncertain(item)
            or not _slot_has_direct_visual_source(item, step_list)
            or not _slot_has_executed_source(item, step_list)
        ):
            continue
        if not _supports(f"{item.get('label', '')} {item.get('value', '')}"):
            continue
        sources = _slot_source_tools(item).intersection(direct_tools)
        source = sorted(sources)[0] if sources else "visual"
        # Structured slots copied from a native result are the same
        # observation, not a second view. Count the executable record below
        # once so duplicate Observer updates cannot satisfy redundancy.
        if source in {"Video_Clip_QA", "Action_Recognition", "Cross_Modal_Verify"}:
            continue
        citation = " ".join(
            str(item.get(key) or "").strip().lower()
            for key in ("citation", "time_window", "supporting_citations")
            if item.get(key) not in (None, "", [], {})
        )
        records.add((
            "slot",
            source,
            citation,
            str(item.get("target") or "").strip().lower(),
            " ".join(str(item.get("observed_value") or item.get("value") or "").lower().split()),
        ))
        timestamp_tokens = sorted(set(re.findall(
            r"(?<![\d.])\d+(?:\.\d+)?\s*(?:s|sec|secs|seconds)\b",
            citation,
            re.IGNORECASE,
        )))
        if len(timestamp_tokens) >= 2:
            # A single packed Inspect_Frame claim can contain several
            # independently sampled timestamps. Preserve that evidence
            # diversity without counting duplicate Observer copies.
            records.add((
                "slot_timestamp",
                source,
                timestamp_tokens[0],
                str(item.get("target") or "").strip().lower(),
                " ".join(str(item.get("observed_value") or item.get("value") or "").lower().split()),
            ))
            records.add((
                "slot_timestamp",
                source,
                timestamp_tokens[-1],
                str(item.get("target") or "").strip().lower(),
                " ".join(str(item.get("observed_value") or item.get("value") or "").lower().split()),
            ))

    for item in _iter_native_observations(step_list):
        answers, texts = _native_option_fields(item)
        evidence = " ".join(answers + texts)
        if not _supports(evidence):
            continue
        ranges = " ".join(
            str(item.get(key) or "").strip().lower()
            for key in (
                "effective_time_range", "requested_time_range", "time_range",
                "search_window", "t_start", "t_end",
            )
            if item.get(key) not in (None, "", [], {})
        )
        query_text = " ".join(
            str(item.get(key) or "").strip().lower()
            for key in ("query", "description")
            if item.get(key) not in (None, "")
        )
        records.add((
            "native",
            _canonical_source_tool(item.get("tool")),
            ranges or query_text,
            "",
            " ".join(str(evidence).lower().split()),
        ))
    return len(records) >= 2


def _has_direct_frame_resolution(
    *,
    answer: str,
    query: str,
    verified_slots: Iterable[Mapping[str, Any]],
    steps: Iterable[Any],
) -> bool:
    """Recognize a later direct-frame observation that resolves old text conflicts.

    Retrieval and clip-QA summaries can be wrong for a short inserted event.  A
    later successful ``Inspect_Frame`` observation is stronger when the ledger
    explicitly cites those frames for the selected answer.  This gate does not
    trust a frame path alone: both the executed frame observation and a verified
    answer-supporting ledger slot are required.
    """
    has_frame_observation = False
    for step in steps:
        if isinstance(step, Mapping):
            action = str(step.get("action") or "").strip()
            items = step.get("observation_items") or []
        else:
            action = str(getattr(step, "action", "") or "").strip()
            items = getattr(step, "observation_items", None) or []
        has_inspect_frame_item = any(
            _canonical_source_tool(item.get("tool")) == "Inspect_Frame"
            for item in _iter_observation_items(items)
            if isinstance(item, Mapping)
        )
        if action != "Inspect_Frame" and not has_inspect_frame_item:
            continue
        for item in _iter_observation_items(items):
            if str(item.get("status") or "").lower() == "error":
                continue
            if item.get("image_evidence"):
                has_frame_observation = True
                break
        if has_frame_observation:
            break
    if not has_frame_observation:
        return False

    answer_tokens = _content_tokens(answer)
    answer_label_match = re.search(r"\(([A-Z])\)", str(answer or "").upper())
    answer_label = answer_label_match.group(1) if answer_label_match else ""
    options = {
        label: text
        for label, text in _OPTION_LINE_RE.findall(str(query or ""))
    }
    selected_option_tokens = _content_tokens(options.get(answer_label, answer))
    for item in verified_slots:
        if bool(item.get("logical conflict", item.get("logical_conflict", False))):
            continue
        citation = str(item.get("citation") or "").lower()
        # The planner often cites a claim as ``... and visual evidence``
        # instead of preserving the concrete tool name.  Once an actual
        # Inspect_Frame observation with image evidence is present, that
        # wording is still a direct-frame citation; requiring the literal
        # tool name makes the safety gate reject a genuinely grounded answer.
        if not any(
            marker in citation
            for marker in (
                "inspect_frame",
                "visual evidence",
                "attached frame",
                "frame evidence",
            )
        ) and not (
            re.search(r"\bframes?(?:[_\-\s]|\b)", citation)
            or re.search(r"\bpack_\d+", citation)
        ):
            continue
        evidence_tokens = _content_tokens(f"{item.get('label', '')} {item.get('value', '')}")
        if answer_tokens.intersection(evidence_tokens):
            return True
        if _dynamic_transfer_paraphrase_match(selected_option_tokens, evidence_tokens):
            return True
    return False


def _has_native_conflict_resolution(
    *,
    answer: str,
    query: str,
    verified_slots: Iterable[Mapping[str, Any]],
    steps: Iterable[Any],
) -> bool:
    """Allow a native answer to supersede stale conflicts after a second view.

    A successful native QA result is still fallible, so it cannot clear a
    conflict on its own.  When the same selected option is supported by a
    native answer and an independently executed direct visual/native view,
    stale retrieval claims may be removed from the active contradiction score.
    The raw trajectory keeps those claims for audit.
    """
    options = {
        label: text
        for label, text in _OPTION_LINE_RE.findall(str(query or ""))
    }
    answer_label_match = re.search(r"\(([A-Z])\)", str(answer or "").upper())
    answer_label = answer_label_match.group(1) if answer_label_match else ""
    if not options or answer_label not in options:
        return False
    native_items = list(_iter_native_observations(steps))
    if not native_items or not _has_independent_evidence(steps=steps):
        return False
    # The native answer must itself support the selected option.  Supplying
    # verified Planner slots is intentionally omitted from this call so a
    # self-reported claim cannot bootstrap conflict resolution.
    if not _has_direct_answer_support(
        answer,
        (),
        query=query,
        steps=steps,
    ):
        return False
    selected_tokens = _content_tokens(options.get(answer_label, ""))
    generic_tokens = _DYNAMIC_GENERIC_TOKENS.union({
        "person", "people", "thing", "object", "item", "one", "same",
        "different", "answer", "option", "choice", "yes", "no",
    })

    def _supports(option_text: str, answer_text: str) -> bool:
        option_tokens = _content_tokens(option_text)
        evidence_tokens = _content_tokens(answer_text)
        specific = option_tokens.intersection(evidence_tokens) - generic_tokens
        if len(option_tokens) <= 2:
            return bool(specific or option_tokens.intersection(evidence_tokens))
        return len(specific) >= 2 or any(len(token) >= 6 for token in specific)

    selected_native_support = False
    competing_native_support = False
    for item in native_items:
        answers, _texts = _native_option_fields(item)
        for answer_text in answers:
            normalized = " ".join(str(answer_text or "").lower().split())
            if not normalized or any(
                marker in normalized
                for marker in ("unknown", "uncertain", "unclear", "maybe", "possibly")
            ):
                continue
            if _supports(options.get(answer_label, ""), normalized):
                selected_native_support = True
            for label, option_text in options.items():
                if label == answer_label:
                    continue
                if _supports(option_text, normalized):
                    competing_native_support = True
    return selected_native_support and not competing_native_support


def _is_prior_textual_conflict(item: Mapping[str, Any]) -> bool:
    """Return true for a stale retrieval/QA conflict superseded by direct frames."""
    if not bool(item.get("logical conflict", item.get("logical_conflict", False))):
        return False
    if str(item.get("evidence_status") or "").lower() != "verified":
        return True
    text = " ".join(
        str(item.get(key) or "")
        for key in ("label", "value", "citation")
    ).lower()
    if "planner thought" in text or "recent_trajectory" in text:
        return True
    if "pack_" in text and "frame_" not in text:
        return True
    if "inspect_frame" in text and not any(
        marker in text
        for marker in ("video_clip_qa", "semantic_retrieve", "segment_retrieve", "retrieval")
    ):
        return False
    return any(
        marker in text
        for marker in (
            "video_clip_qa",
            "semantic_retrieve",
            "segment_retrieve",
            "matched_segments",
            "retrieval",
            "clean_results",
        )
    )


def _answer_matches_selected_option(
    *,
    answer: str,
    answer_label: str,
    options: Mapping[str, str],
) -> bool:
    """Reject ``(D) 7`` when the listed D option is ``1``.

    A tool may return a real observation that is outside the benchmark's
    answer choices.  The planner still has to select the exact listed option;
    a matching label alone is not enough.
    """

    if not answer_label or answer_label not in options:
        return False
    match = re.search(r"\([A-Z]\)\s*(.*?)\s*$", str(answer or ""), re.IGNORECASE | re.DOTALL)
    answer_text = str(match.group(1) if match else "").strip()
    option_text = str(options.get(answer_label) or "").strip()
    if not answer_text or not option_text:
        return False
    answer_tokens = _content_tokens(answer_text)
    option_tokens = _content_tokens(option_text)
    return bool(option_tokens) and option_tokens.issubset(answer_tokens)


def _has_order_answer_support(
    *,
    answer_label: str,
    options: Mapping[str, str],
    query: str,
    verified_slots: Iterable[Mapping[str, Any]],
    steps: Iterable[Any] = (),
) -> bool:
    """Recognize evidence that states the event order behind an MLVU option.

    Order answers are encoded as number sequences, while the evidence naturally
    describes the events.  Requiring literal token overlap therefore rejects a
    correctly grounded answer such as ``hand taping -> boxing -> sit-ups ->
    bikini photos`` for option ``(D) 1->2->3->4``.
    """
    option_sequence = re.findall(r"\d+", str(options.get(answer_label, "")))
    if not option_sequence:
        return False
    event_text_by_number = {
        number: text
        for number, text in re.findall(r"\((\d+)\)\s*([^;\n]+)", str(query or ""))
    }
    if not all(number in event_text_by_number for number in option_sequence):
        return False
    # A retrieval sentence can contain an apparently ordered list while
    # carrying no executable visual observation. Keep order support grounded in
    # a direct visual/native claim; retrieval remains useful for routing only.
    direct_verified_slots = [
        item for item in verified_slots
        if isinstance(item, Mapping)
        and _slot_has_direct_visual_source(item, steps)
        and not bool(item.get("context_only", False))
        and not bool(item.get("superseded", False))
        and not _slot_is_uncertain(item)
    ]

    event_times: dict[str, float] = {}
    for item in direct_verified_slots:
        label_match = re.search(
            r"\bevent(?:[_\s]*\(?\s*)(\d+)\s*\)?",
            str(item.get("label", "")),
            re.I,
        )
        if not label_match:
            continue
        temporal_text = f"{item.get('value', '')} {item.get('citation', '')}"
        time_match = re.search(
            r"(?:timestamp|at|occurs?\s+at|from)\s*[~@:]?\s*(\d+(?:\.\d+)?)\s*s?",
            temporal_text,
            re.I,
        )
        if time_match:
            event_times[label_match.group(1)] = float(time_match.group(1))
        else:
            value_match = re.fullmatch(
                r"\s*(\d+(?:\.\d+)?)\s*s?\s*",
                str(item.get("value", "")),
                re.I,
            )
            if value_match:
                event_times[label_match.group(1)] = float(value_match.group(1))
    if all(number in event_times for number in option_sequence):
        observed_sequence = sorted(option_sequence, key=event_times.__getitem__)
        return observed_sequence == option_sequence

    for item in direct_verified_slots:
        claim_text = f"{item.get('label', '')} {item.get('value', '')}".lower()
        compact_claim = re.sub(r"\s+", "", claim_text)
        compact_sequence = "->".join(option_sequence)
        if compact_sequence in compact_claim:
            return True

        positions = []
        for number in option_sequence:
            event_tokens = _content_tokens(event_text_by_number[number])
            token_positions = [
                claim_text.find(token)
                for token in event_tokens
                if len(token) >= 3 and claim_text.find(token) >= 0
            ]
            if not token_positions:
                break
            positions.append(min(token_positions))
        else:
            if positions == sorted(positions) and (
                "->" in claim_text
                or any(marker in claim_text for marker in ("order", "sequence", "chronological", "then", "followed"))
            ):
                return True

    # A native Video_Clip_QA answer may state the sequence directly while the
    # Observer omits event-number slots.  Accept only an explicit numeric
    # chain or a complete ordered set of event descriptors from that native
    # result; a bare word such as ``chronological`` is insufficient.
    event_texts = {
        number: " ".join(text.split()).lower()
        for number, text in event_text_by_number.items()
    }
    for native_text in _native_answer_texts(steps):
        claim_text = " ".join(native_text.lower().split())
        compact = re.sub(r"\s+", "", claim_text).replace("→", "->")
        if "->".join(option_sequence) in compact:
            return True
        positions: List[int] = []
        for number in option_sequence:
            event_tokens = _content_tokens(event_texts[number])
            candidate_positions = [
                claim_text.find(token)
                for token in event_tokens
                if len(token) >= 3 and claim_text.find(token) >= 0
            ]
            if not candidate_positions:
                break
            positions.append(min(candidate_positions))
        else:
            if positions == sorted(positions) and any(
                marker in claim_text
                for marker in ("order", "sequence", "chronological", "then", "followed", "first", "next")
            ):
                return True
    return False


def _observed_order_sequence(
    *,
    query: str,
    verified_slots: Iterable[Mapping[str, Any]],
) -> List[str]:
    """Extract a verified event order from event timestamps or an explicit chain."""
    event_numbers = [
        number
        for number, _text in re.findall(
            r"\((\d+)\)\s*([^;\n]+)", str(query or "")
        )
    ]
    if not event_numbers:
        return []
    event_times: dict[str, float] = {}
    for item in verified_slots:
        label_match = re.search(
            r"\bevent(?:[_\s]*\(?\s*)(\d+)\s*\)?",
            str(item.get("label", "")),
            re.I,
        )
        if not label_match:
            continue
        temporal_text = f"{item.get('value', '')} {item.get('citation', '')}"
        time_match = re.search(
            r"(?:timestamp|at|occurs?\s+at|from)\s*[~@:]?\s*(\d+(?:\.\d+)?)\s*s?",
            temporal_text,
            re.I,
        )
        if time_match:
            event_times[label_match.group(1)] = float(time_match.group(1))
        else:
            value_match = re.fullmatch(
                r"\s*(\d+(?:\.\d+)?)\s*s?\s*",
                str(item.get("value", "")),
                re.I,
            )
            if value_match:
                event_times[label_match.group(1)] = float(value_match.group(1))
    if all(number in event_times for number in event_numbers):
        return sorted(event_numbers, key=event_times.__getitem__)

    required = set(event_numbers)
    for item in verified_slots:
        claim_text = " ".join(
            str(item.get(key, "") or "") for key in ("label", "value", "citation")
        )
        for chain in re.findall(r"\d+(?:\s*->\s*\d+)+", claim_text):
            sequence = re.findall(r"\d+", chain)
            if set(sequence) >= required:
                return sequence
    return []


def _verified_order_has_no_listed_option(
    *,
    query: str,
    options: Mapping[str, str],
    verified_slots: Iterable[Mapping[str, Any]],
) -> bool:
    """Detect a grounded order that cannot be represented by any answer option."""
    observed = _observed_order_sequence(query=query, verified_slots=verified_slots)
    if not observed:
        return False
    listed_sequences = {
        tuple(re.findall(r"\d+", str(option_text or "")))
        for option_text in options.values()
        if re.findall(r"\d+", str(option_text or ""))
    }
    return tuple(observed) not in listed_sequences


def _iter_observation_items(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_observation_items(item)
        return
    if not isinstance(value, Mapping):
        return
    if "status" in value and str(value.get("status") or "").strip().lower() != "success":
        return
    nested = value.get("items")
    if isinstance(nested, list):
        yield from _iter_observation_items(nested)
    probes = value.get("probes")
    if isinstance(probes, list) and str(value.get("tool") or "").strip():
        for probe in probes:
            if isinstance(probe, Mapping):
                yield from _iter_observation_items({
                    "tool": value["tool"], "status": value.get("status", "success"), **probe})
    if str(value.get("tool") or "").strip():
        yield value


def _has_dynamic_answer_support(
    answer: str,
    steps: Iterable[Any],
    verified_slots: Iterable[Mapping[str, Any]],
    query: str = "",
) -> bool:
    steps = list(steps)
    verified_slots = list(verified_slots)
    answer_tokens = _content_tokens(answer)
    required_overlap = 1 if len(answer_tokens) <= 2 else 2
    answer_label_match = re.search(r"\(([A-Z])\)", str(answer or "").upper())
    answer_label = answer_label_match.group(1) if answer_label_match else ""
    options = {
        label: text
        for label, text in _OPTION_LINE_RE.findall(str(query or ""))
    }
    selected_option_tokens = _content_tokens(options.get(answer_label, answer))

    dynamic_result_tokens: List[set[str]] = []
    for step in steps:
        if isinstance(step, Mapping):
            items = step.get("observation_items") or []
        else:
            items = getattr(step, "observation_items", None) or []
        for item in _iter_observation_items(items):
            if _canonical_source_tool(item.get("tool")) not in {"Video_Clip_QA", "Action_Recognition", "Cross_Modal_Verify"}:
                continue
            status = str(item.get("status") or "").lower()
            if status != "success":
                continue
            result_tokens = _content_tokens(item.get("result"))
            dynamic_result_tokens.append(result_tokens)
            if len(answer_tokens.intersection(result_tokens)) >= required_overlap:
                return True
            if _dynamic_transfer_paraphrase_match(selected_option_tokens, result_tokens):
                return True
    # Timestamped direct visual observations can establish an action/relation
    # without requiring one particular QA backend. Counts still need their
    # dedicated count/coverage contract rather than two arbitrary frames.
    if not _query_asks_action_count(_semantic_query_text(query)) and _has_independent_evidence(steps=steps):
        for item in verified_slots:
            if (str(item.get("evidence_status") or "").lower() == "verified"
                    and not _slot_is_uncertain(item) and not _slot_is_synthetic_only(item)
                    and not item.get("superseded")
                    and not item.get("logical_conflict", item.get("logical conflict", False))
                    and _slot_has_direct_visual_source(item, steps) and _slot_has_executed_source(item, steps)
                    and (item.get("relation") or _has_dynamic_relation(_claim_contract_text(item)))):
                # Observer may put the action in label/relation and the
                # measurement in observed_value. Read the verified semantic
                # fields together, excluding citation and requested qualifiers.
                tokens = _content_tokens(" ".join(str(item.get(key) or "") for key in
                    ("label", "target", "relation", "observed_value", "value")))
                if selected_option_tokens and all(
                    token in tokens or _CONTENT_TOKEN_EQUIVALENTS.get(token, frozenset()).intersection(tokens)
                    for token in selected_option_tokens
                ):
                    return True
    if not dynamic_result_tokens:
        return False
    for item in verified_slots:
        if not _slot_source_tools(item).intersection({"Video_Clip_QA", "Action_Recognition", "Cross_Modal_Verify"}):
            continue
        claim_tokens = _content_tokens(f"{item.get('label', '')} {item.get('value', '')}")
        if answer_tokens.intersection(claim_tokens) and any(
            len({
                token for token in claim_tokens.intersection(result_tokens)
                if token not in _DYNAMIC_GENERIC_TOKENS and not token.isdigit()
            }) >= 2
            for result_tokens in dynamic_result_tokens
        ):
            return True
        if _dynamic_transfer_paraphrase_match(selected_option_tokens, claim_tokens) and any(
            len({
                token for token in claim_tokens.intersection(result_tokens)
                if token not in _DYNAMIC_GENERIC_TOKENS and not token.isdigit()
            }) >= 1
            for result_tokens in dynamic_result_tokens
        ):
            return True
    return False


def _action_total_semantic_mismatch(
    *,
    query: str,
    verified_slots: Iterable[Mapping[str, Any]],
) -> bool:
    """Reject total-count evidence when the question asks for an action count.

    A common visual QA failure is to observe one object being added to a
    container, notice two objects already present, and answer the resulting
    total.  The evidence may be internally consistent while answering a
    different question.  This gate is deliberately conservative: it only
    fires when the query is a count question with an action verb and the
    supporting claim contains total/pre-existing context without an explicit
    action-count label or direct action-count phrase.
    """
    normalized_query = _semantic_query_text(query)
    if not any(marker in normalized_query for marker in _COUNT_QUERY_MARKERS):
        return False
    query_tokens = set(re.findall(r"[a-z]+", normalized_query))
    if not query_tokens.intersection(_ACTION_QUERY_MARKERS):
        return False
    verified_items = [item for item in verified_slots if isinstance(item, Mapping)]
    has_verified_direct_action_relation = any(
        not bool(item.get("context_only", False))
        and _DIRECT_ACTION_COUNT_RE.search(
            " ".join(
                str(item.get(key) or "")
                for key in ("label", "value", "citation")
            )
        )
        and _direct_action_relation_support(
            query=normalized_query,
            claim=" ".join(
                str(item.get(key) or "")
                for key in ("label", "value", "citation")
            ),
        )
        for item in verified_items
    )
    has_verified_action_scope = any(
        str(item.get("scope") or "").strip().lower() == "action_count"
        and not bool(item.get("context_only", False))
        and _direct_action_relation_support(
            query=normalized_query,
            claim=" ".join(
                str(item.get(key) or "")
                for key in ("label", "value", "citation")
            ),
        )
        for item in verified_items
    )
    has_verified_action_evidence = (
        has_verified_action_scope or has_verified_direct_action_relation
    )
    for item in verified_items:
        # Once a directly grounded action-count slot exists, a ledger entry
        # explicitly marked context-only describes the pre-existing/total
        # state and is not a semantic mismatch. If no action slot exists,
        # context-only total evidence must still be rejected.
        if bool(item.get("context_only", False)) and has_verified_action_evidence:
            continue
        explicit_scope = str(item.get("scope") or "").strip().lower()
        if has_verified_action_evidence and explicit_scope in {
            "total_count",
            "state_count",
            "scene_count",
            "generic_count",
        }:
            # The action slot is the answer-bearing scope for an action
            # question.  Keep total/state/generic observations available for
            # audit, but do not let them block a directly grounded action
            # count if an older trajectory did not pass through the planner's
            # ledger demotion step.
            continue
        if (
            explicit_scope in {"total_count", "state_count", "scene_count"}
            and not bool(item.get("context_only", False))
            and not has_verified_action_evidence
        ):
            return True
        label = str(item.get("label") or "").lower()
        claim = " ".join(
            str(item.get(key) or "").lower()
            for key in ("label", "value", "citation")
        )
        claim_tokens = set(re.findall(r"[a-z]+", claim))
        if not claim_tokens.intersection(_TOTAL_CONTEXT_MARKERS):
            continue
        if (
            any(marker in label for marker in ("action", "placed", "put", "added", "removed", "inserted"))
            and _direct_action_relation_support(query=normalized_query, claim=claim)
        ):
            continue
        if _DIRECT_ACTION_COUNT_RE.search(claim) and _direct_action_relation_support(
            query=normalized_query,
            claim=claim,
        ):
            continue
        return True
    return False


def _query_asks_action_count(query: str) -> bool:
    normalized = " ".join(str(query or "").lower().split())
    if not any(marker in normalized for marker in ("how many", "number of", "count")):
        return False
    tokens = set(re.findall(r"[a-z]+", normalized))
    if tokens.intersection(_ACTION_QUERY_MARKERS):
        return True
    return bool(re.search(
        r"\b(?:instance|instances|occurrence|occurrences)\b[^.?!]{0,60}\b"
        r"(?:action|actions|event|events|scene|scenes)\b",
        normalized,
    ))


def _query_asks_purpose(query: str) -> bool:
    normalized = " ".join(str(query or "").lower().split())
    return any(marker in normalized for marker in _PURPOSE_QUERY_MARKERS) or bool(
        re.search(r"\b(?:contribute|contributes|quality)\b", normalized)
    )


def _query_asks_rationale(query: str) -> bool:
    """Return whether the question asks for a reason, importance, or inference.

    These questions need a causal/effect statement.  An observed outcome such
    as ``the cut is straight`` is not enough to answer ``why precision is
    important``; the ledger must preserve the link (``precision ensures a
    straight cut``) or a native QA result must state it directly.
    """
    normalized = " ".join(str(query or "").lower().split())
    if any(marker in normalized for marker in _PURPOSE_QUERY_MARKERS):
        return True
    if re.search(r"\b(?:why|importance|important|significance|significant|reason)\b", normalized):
        return True
    if re.search(
        r"\b(?:what\s+can\s+be\s+inferred|how\s+.*\b(?:contribute|demonstrate|demonstrated|impact))\b",
        normalized,
    ):
        return True
    if re.search(r"\b(?:contribute|contributes|quality|demonstrates?)\b", normalized):
        return True
    return False


def _semantic_query_text(query: str) -> str:
    """Strip task/options/output protocol text before semantic gate checks."""
    raw_query = str(query or "")
    question_match = re.search(r"(?is)\bquestion\s*:\s*(.*)", raw_query)
    semantic_query = question_match.group(1) if question_match else raw_query
    semantic_query = re.split(
        r"(?im)^\s*(?:options?|choices?)\s*:\s*", semantic_query, maxsplit=1
    )[0]
    semantic_query = re.split(
        r"(?im)^\s*reply\s+with\b", semantic_query, maxsplit=1
    )[0]
    # Prompts may append prose-style instructions such as "without
    # mentioning/listing actions".  These are not visual negations or
    # temporal qualifiers; strip only the explicit meta forms while retaining
    # factual phrases such as "without paying".
    semantic_query = re.sub(
        r"\bwithout\s+(?:just\s+|simply\s+)?(?:mention(?:ing)?|list(?:ing)?|"
        r"restat(?:ing|e)|repeating|enumerat(?:ing|e))\b[^.?!\n]*",
        " ",
        semantic_query,
        flags=re.IGNORECASE,
    )
    semantic_query = re.split(r"(?m)^\s*\([A-Z]\)\s+", semantic_query, maxsplit=1)[0]
    semantic_query = re.sub(
        r"(?im)^\s*[^\n:]*\b(?:task\s*type|dataset)\s*:[^\n]*",
        "", semantic_query,
    )
    return " ".join(semantic_query.lower().split())


def _iter_native_observations(steps: Iterable[Any]) -> Iterable[Mapping[str, Any]]:
    """Yield successful executable native visual/relation observations only."""
    native_tools = {"Video_Clip_QA", "Action_Recognition", "Cross_Modal_Verify"}
    for step in steps:
        if isinstance(step, Mapping):
            observation_items = step.get("observation_items") or []
        else:
            observation_items = getattr(step, "observation_items", None) or []
        for item in _iter_observation_items(observation_items):
            if _canonical_source_tool(item.get("tool")) not in native_tools:
                continue
            if str(item.get("status") or "").strip().lower() != "success":
                continue
            yield item


def _native_observation_text(item: Mapping[str, Any]) -> str:
    """Collect answer/evidence fields from a native tool without trusting prose slots."""
    parts: List[str] = []
    payload = _coerce_mapping(item.get("result"))
    parsed = _coerce_mapping(item.get("parsed_json"))
    if not parsed:
        parsed = _coerce_mapping(payload.get("parsed_json"))
    sources = [parsed, payload, item]
    for source in sources:
        for key in (
            "answer", "answer_value", "evidence", "evidence_items", "video_summary",
            "video_summary_value", "summary", "result",
        ):
            value = source.get(key) if isinstance(source, Mapping) else None
            if isinstance(value, Mapping):
                value = value.get("answer") or value.get("value") or value.get("text")
            if isinstance(value, (list, tuple, set)):
                parts.extend(str(entry) for entry in value)
            elif value not in (None, ""):
                parts.append(str(value))
    return " ".join(parts)


def _native_answer_texts(steps: Iterable[Any]) -> Iterable[str]:
    for item in _iter_native_observations(steps):
        text = _native_observation_text(item).strip()
        if text:
            yield text


def _has_native_rationale_support(steps: Iterable[Any]) -> bool:
    """Require a native causal/use statement for why/importance questions."""
    causal_markers = (
        "because", "ensure", "ensures", "so that", "in order to", "purpose",
        "function", "used for", "used to", "serves to", "helps", "allows",
        "reason is", "reason:",
    )
    for text in _native_answer_texts(steps):
        normalized = " ".join(text.lower().split())
        if any(marker in normalized for marker in causal_markers):
            # A causal marker must connect to a concrete outcome rather than
            # occur in a generic restatement of the question.
            concrete = _content_tokens(normalized) - _content_tokens(
                "because ensure ensures purpose function reason action result goal object important"
            )
            if len(concrete) >= 2 and not _slot_is_uncertain({"uncertainty": normalized}):
                return True
    return False


def _rationale_option_is_ambiguous(
    *,
    answer_label: str,
    options: Mapping[str, str],
    verified_slots: Iterable[Mapping[str, Any]],
    steps: Iterable[Any],
    query: str,
) -> bool:
    """Detect when evidence supports multiple rationale-option attributes.

    Rationale questions often present near-identical choices that differ only
    in the claimed outcome (for example ``straight`` versus ``correct size``).
    Matching one word in the selected option is unsafe when a second option is
    supported by the same direct observation.  Require an explicit native
    discriminator or leave the candidate unresolved for another probe.
    """

    if not answer_label or answer_label not in options or not _query_asks_rationale(query):
        return False
    if len(options) < 2:
        return False
    option_tokens = {
        label: _content_tokens(text)
        for label, text in options.items()
    }
    token_frequency = Counter(
        token
        for tokens in option_tokens.values()
        for token in tokens
    )
    generic = {
        "character", "action", "actions", "important", "importance", "ensure",
        "ensures", "process", "thing", "person", "people", "one",
    }
    specific_tokens = {
        label: {
            token for token in tokens
            if token not in generic and token_frequency.get(token, 0) <= max(1, len(options) // 2)
        }
        for label, tokens in option_tokens.items()
    }
    evidence_tokens: set[str] = set()
    for item in verified_slots:
        if not isinstance(item, Mapping) or _slot_is_uncertain(item) or _slot_is_synthetic_only(item):
            continue
        slot_text = " ".join(
            str(item.get(key, "") or "")
            for key in ("observed_value", "value")
        )
        # Direction is an outcome for attribute/state claims (``straight,
        # even``), but for a purpose/action slot it often repeats the input
        # object (``into the cup of water``).  Do not let that input noun
        # create a false second option match.
        relation = str(item.get("relation") or "").strip().lower()
        if relation in {"is", "has", "equals", "compared", "different", "same"}:
            slot_text = f"{slot_text} {item.get('direction', '')}"
        evidence_tokens.update(_content_tokens(slot_text))
    # A native action description is not itself a purpose/outcome claim. Only
    # add native text when it contains an explicit causal/use relation; the
    # ordinary action sentence ``dips the brush into water`` must not make the
    # ``source of water`` option look supported.
    if _has_native_rationale_support(steps):
        for text in _native_answer_texts(steps):
            evidence_tokens.update(_content_tokens(text))
    def _supported(option_specific: set[str]) -> bool:
        return bool(
            option_specific.intersection(evidence_tokens)
            or any(
                _CONTENT_TOKEN_EQUIVALENTS.get(token, frozenset()).intersection(evidence_tokens)
                for token in option_specific
            )
        )

    supported = {
        label for label, tokens in specific_tokens.items()
        if tokens and _supported(tokens)
    }
    if len(supported) < 2 or answer_label not in supported:
        return False

    # A native answer that explicitly disambiguates alternatives can resolve
    # the overlap; a bare categorical answer or a list of two attributes
    # cannot.  Keep this narrow so ordinary multi-claim evidence remains
    # usable when the tool clearly states the comparison.
    comparison_text = " ".join(_native_answer_texts(steps)).lower()
    explicit_discriminator = bool(re.search(
        r"\b(?:only|rather than|instead of|specifically|the answer is|option\s+[a-z])\b",
        comparison_text,
    ))
    return not explicit_discriminator


def _has_query_qualifier_support(
    *,
    query: str,
    verified_slots: Iterable[Mapping[str, Any]],
    steps: Iterable[Any],
) -> bool:
    """Check temporal/spatial/negation qualifiers without literal-only traps."""
    normalized_query = _semantic_query_text(query)
    qualifiers = [
        marker for marker in (
            "exactly", "at least", "at most", "only", "first", "last", "immediately",
            "before", "after", "left", "right", "directly", "without", "never",
        )
        if re.search(rf"\b{re.escape(marker)}\b", normalized_query)
    ]
    if not qualifiers:
        return True
    slot_text = " ".join(
        _claim_contract_text(item)
        for item in verified_slots
        if isinstance(item, Mapping)
    )
    native_text_parts = list(_native_answer_texts(steps))
    for step in steps:
        observations = (
            step.get("observation_items") or []
            if isinstance(step, Mapping)
            else getattr(step, "observation_items", None) or []
        )
        for item in _iter_observation_items(observations):
            if _canonical_source_tool(item.get("tool")) != "Video_Summary":
                continue
            if str(item.get("status") or "").strip().lower() != "success":
                continue
            text = _native_observation_text(item).strip()
            if text:
                native_text_parts.append(text)
    native_text = " ".join(native_text_parts).lower()
    combined = f"{slot_text} {native_text}"
    for marker in qualifiers:
        if marker in combined:
            continue
        if marker in {"before", "after"} and any(
            token in combined
            for token in ("earlier", "later", "then", "subsequent", "following", "preceding", "before", "after")
        ):
            continue
        if marker in {"first", "last"} and any(
            token in combined
            for token in ("first", "last", "earliest", "latest", "initial", "final", "chronological", "sequence")
        ):
            continue
        if marker in {"left", "right"} and any(
            token in combined
            for token in ("left", "right", "direction", "beside", "next to")
        ):
            continue
        if marker in {"without", "never"} and any(
            token in combined
            for token in ("absent", "absence", "not present", "not shown", "did not", "does not", "never", "without", "no ")
        ):
            continue
        if marker == "only" and any(
            token in combined
            for token in ("only", "sole", "unique", "exclusive", "none of the others")
        ):
            continue
        return False
    return True


def _native_count_values(steps: Iterable[Any]) -> List[str]:
    """Extract numeric answers from successful native QA records."""
    values: List[str] = []
    for item in _iter_native_observations(steps):
        answers, _texts = _native_option_fields(item)
        for answer in answers:
            queue: List[Any] = [answer]
            seen: set[str] = set()
            while queue:
                current = queue.pop(0)
                if isinstance(current, Mapping):
                    for key in ("answer", "answer_value", "value"):
                        if key in current:
                            queue.append(current.get(key))
                    continue
                text = str(current or "").strip()
                if not text:
                    continue
                if text.startswith(("{", "[")):
                    try:
                        parsed = json.loads(text)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, (Mapping, list)):
                        queue.append(parsed)
                        continue
                normalized = " ".join(text.lower().split())
                if normalized in seen:
                    continue
                seen.add(normalized)
                # Only an answer field is considered; timestamps and segment
                # identifiers in the surrounding result are not count claims.
                match = re.fullmatch(
                    r"(?:there\s+are\s+|a\s+total\s+of\s+|total\s*[:=]?\s*)?"
                    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
                    r"(?:\s+(?:items?|objects?|bridges?|doors?|instances?|times?))?[.!]?",
                    normalized,
                    re.IGNORECASE,
                )
                if not match:
                    continue
                token = re.findall(r"\d+", normalized)
                if token:
                    value = token[0]
                else:
                    word = re.search(
                        r"zero|one|two|three|four|five|six|seven|eight|nine|ten",
                        normalized,
                        re.IGNORECASE,
                    )
                    value = _NUMBER_WORDS.get(str(word.group(0)).lower(), "") if word else ""
                if value and value not in values:
                    values.append(value)
    return values


def _count_evidence_conflict(
    *,
    query: str,
    verified_slots: Iterable[Mapping[str, Any]],
    steps: Iterable[Any],
) -> bool:
    """Detect incompatible direct/native numeric count claims."""
    normalized = " ".join(str(query or "").lower().split())
    if not re.search(r"\b(?:how\s+many|number\s+of|count|total)\b", normalized):
        return False
    native_values = set(_native_count_values(steps))
    slot_values: set[str] = set()
    for item in verified_slots:
        if not isinstance(item, Mapping) or _slot_is_synthetic_only(item):
            continue
        scope = str(item.get("scope") or "").strip().lower()
        if scope not in {"action_count", "total_count", "state_count", "scene_count", "generic_count", "comparison"}:
            continue
        text = " ".join(
            str(item.get(key) or "") for key in ("observed_value", "value")
        )
        numbers = re.findall(r"(?<![a-z])(\d+)(?![a-z])", text.lower())
        for number in numbers:
            if number not in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}:
                continue
            slot_values.add(number)
    if len(native_values) >= 2:
        return True
    return bool(native_values and slot_values and not native_values.intersection(slot_values))


def _incompatible_verified_claims(
    *,
    query: str,
    verified_slots: Iterable[Mapping[str, Any]],
    steps: Iterable[Any],
) -> bool:
    """Reject mutually exclusive direct attributes for one identified target.

    Observer claims can be individually well formed while still describing the
    same person/object with incompatible values. Logical-conflict flags are
    optional, so the verifier must also compare executed visual claims itself.
    The check is limited to attribute/identity questions and only treats claims
    observed in overlapping or unknown windows as conflicting; values observed
    in disjoint windows may legitimately change over time.
    """
    semantic = _semantic_query_text(query)
    if not re.search(
        r"\b(?:color|colour|hair|eye|eyes|weather|instrument|object|surface|"
        r"shape|appearance|perspective|position|location|vehicle|profession|"
        r"occupation|identity|age|gender|clothes|clothing|shirt|dress|"
        r"teeth|background|purpose|function|prepare|preparing|process|"
        r"quality|contribute|contributes|depending|differently|"
        r"what\s+part|which\s+item|who)\b",
        semantic,
        re.IGNORECASE,
    ):
        return False

    direct_tools = {
        "Inspect_Frame", "Inspect_Clip", "Crop", "Zoom_in",
        "Video_Clip_QA", "Action_Recognition", "Cross_Modal_Verify",
    }
    step_list = list(steps)
    active_tools: set[str] = set()
    for step in step_list:
        observations = (
            step.get("observation_items") or []
            if isinstance(step, Mapping)
            else getattr(step, "observation_items", None) or []
        )
        for observation in _iter_observation_items(observations):
            tool = _canonical_source_tool(observation.get("tool"))
            status = str(observation.get("status") or "").strip().lower()
            if tool in direct_tools and status not in {"error", "failed", "no_valid_result"}:
                active_tools.add(tool)
    if not active_tools:
        return False

    generic_target_tokens = {
        "a", "an", "the", "one", "two", "both", "each", "all", "kid", "kids",
        "child", "children", "person", "people", "man", "woman", "men", "women",
        "figure", "character", "object", "item", "thing", "subject", "video",
        "scene", "frame", "left", "right", "masked", "unmasked",
    }
    generic_value_tokens = {
        "a", "an", "the", "one", "two", "both", "each", "all", "has", "have",
        "is", "are", "was", "were", "hair", "color", "colour", "eye", "eyes",
        "instrument", "object", "item", "person", "people", "kid", "kids",
        "child", "children", "figure", "character", "visible", "shown", "present",
        "appears", "appearing", "looks", "seems", "none", "unknown", "attribute",
        "state", "value", "same", "different",
    }

    def _target_key(item: Mapping[str, Any]) -> str:
        raw = str(item.get("target") or "").strip().lower()
        if not raw:
            return ""
        # Parenthetical role descriptions are not stable identities.
        raw = re.sub(r"\([^)]*\)", "", raw)
        tokens = re.findall(r"[a-z0-9]+", raw)
        if not tokens:
            return ""
        compact = "".join(tokens)
        if not compact or set(tokens).issubset(generic_target_tokens):
            return ""
        # kid_1 and kid1 should canonicalize identically.
        compact = re.sub(r"^(?:the)?(?:kid|child|figure|person)", "", compact)
        compact = re.sub(r"(?:dipping|interaction|action|purpose|event)$", "", compact)
        return compact or ""

    def _value_tokens(item: Mapping[str, Any]) -> set[str]:
        raw = item.get("observed_value")
        if raw in (None, ""):
            raw = item.get("value")
        tokens = _content_tokens(raw)
        return {
            token for token in tokens
            if token not in generic_value_tokens
            and token not in {"none", "null", "n"}
        }

    def _claim_ranges(item: Mapping[str, Any]) -> List[tuple[float, float]]:
        ranges: List[tuple[float, float]] = []
        values: List[Any] = []
        for key in (
            "time_window", "requested_time_range", "effective_time_range",
            "timestamp_range", "search_window",
        ):
            if item.get(key) not in (None, "", [], {}):
                values.append(item.get(key))
        values.extend(
            item.get(key) for key in ("citation", "supporting_citations")
            if item.get(key) not in (None, "", [], {})
        )
        index = 0
        while index < len(values):
            value = values[index]
            index += 1
            if isinstance(value, (list, tuple, set)):
                values.extend(value)
                continue
            if isinstance(value, Mapping):
                for start_key, end_key in (
                    ("start", "end"), ("start_time", "end_time"),
                    ("t_start", "t_end"),
                ):
                    if value.get(start_key) is None or value.get(end_key) is None:
                        continue
                    try:
                        start, end = float(value[start_key]), float(value[end_key])
                    except (TypeError, ValueError):
                        continue
                    if end >= start:
                        ranges.append((start, end))
                continue
            text = str(value or "")
            for match in re.finditer(
                r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?\s*"
                r"(?:[-–]|to)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?",
                text,
                re.IGNORECASE,
            ):
                try:
                    start, end = float(match.group(1)), float(match.group(2))
                except (TypeError, ValueError):
                    continue
                if end >= start:
                    ranges.append((start, end))
            for match in re.finditer(
                r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)\b",
                text,
                re.IGNORECASE,
            ):
                try:
                    point = float(match.group(1))
                except (TypeError, ValueError):
                    continue
                ranges.append((point, point))
        return ranges

    def _windows_overlap(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
        first_ranges = _claim_ranges(first)
        second_ranges = _claim_ranges(second)
        if not first_ranges or not second_ranges:
            # Missing timing is unsafe to interpret as disjoint.
            return True
        return any(
            left_start <= right_end + 1e-6
            and right_start <= left_end + 1e-6
            for left_start, left_end in first_ranges
            for right_start, right_end in second_ranges
        )

    grouped: Dict[tuple[str, str], List[Mapping[str, Any]]] = {}
    for item in verified_slots:
        if not isinstance(item, Mapping):
            continue
        if bool(item.get("context_only", False)) or bool(item.get("superseded", False)):
            continue
        if _slot_is_synthetic_only(item) or not _slot_has_direct_visual_source(item, step_list):
            continue
        if not _slot_has_executed_source(item, step_list):
            continue
        uncertainty = str(item.get("uncertainty") or "").strip().lower()
        if any(marker in uncertainty for marker in _UNCERTAINTY_MARKERS) or uncertainty in {
            "low", "medium", "low confidence", "medium confidence",
        }:
            continue
        target = _target_key(item)
        values = _value_tokens(item)
        relation = str(item.get("relation") or "").strip().lower()
        raw_label = "".join(re.findall(r"[a-z0-9]+", str(item.get("label") or "").lower()))
        label = (
            "purpose"
            if relation in {"purpose", "causes", "function", "used_for", "used_to"}
            or "purpose" in raw_label
            else raw_label
        )
        if not target or not label or not values:
            continue
        grouped.setdefault((label, target), []).append(item)

    for claims in grouped.values():
        for index, first in enumerate(claims):
            first_values = _value_tokens(first)
            for second in claims[index + 1:]:
                second_values = _value_tokens(second)
                if first_values.intersection(second_values):
                    continue
                if _windows_overlap(first, second):
                    return True
    return False


def _query_requires_independent_evidence(query: str, thresholds: Mapping[str, Any]) -> bool:
    """Identify query forms where one visual glance is a known failure mode."""
    semantic = _semantic_query_text(query)
    if not _OPTION_LINE_RE.search(str(query or "")):
        return False
    if _requires_option_comparison(query, thresholds):
        return True
    if re.search(
        r"\b(?:how\s+many|number\s+of|count|total|sequence|chronological|order|"
        r"overall\s+(?:theme|objective|goal)|"
        r"throughout|contribute|contributes|importance|significance|significant|"
        r"inferred|deviation|differently|depending|primary\s+objective|"
        r"main\s+methods?|respectively)\b",
        semantic,
        re.IGNORECASE,
    ):
        return True
    return False


def _has_independent_evidence(*, steps: Iterable[Any]) -> bool:
    """Require two materially distinct direct observations for high-risk queries."""
    records: List[tuple[str, str, str, List[float]]] = []
    for item in _iter_observation_items(
        [
            observation
            for step in steps
            for observation in (
                step.get("observation_items") or []
                if isinstance(step, Mapping)
                else getattr(step, "observation_items", None) or []
            )
        ]
    ):
        tool = _canonical_source_tool(item.get("tool"))
        if tool not in {
            "Inspect_Frame", "Inspect_Clip", "Crop", "Zoom_in", "Video_Clip_QA",
            "Action_Recognition", "Cross_Modal_Verify",
        } and not _observation_image_assets(item):
            continue
        if str(item.get("status") or "").strip().lower() in {"error", "no_valid_result", "failed"}:
            continue
        raw_times = item.get("image_timestamps") or item.get("sample_times") or []
        times: List[float] = []
        if isinstance(raw_times, (list, tuple, set)):
            for value in raw_times:
                try:
                    timestamp = float(value)
                    if math.isfinite(timestamp) and timestamp >= 0:
                        times.append(timestamp)
                except (TypeError, ValueError):
                    continue
        times = sorted(set(times))
        ranges: List[str] = []
        for key in ("requested_time_range", "effective_time_range", "time_range", "search_window"):
            value = item.get(key)
            if isinstance(value, Mapping):
                ranges.append(f"{value.get('start', value.get('start_time', ''))}-{value.get('end', value.get('end_time', ''))}")
            elif value not in (None, ""):
                ranges.append(str(value))
        if item.get("t_start") is not None or item.get("t_end") is not None:
            ranges.append(f"{item.get('t_start')}-{item.get('t_end')}")
        query_text = " ".join(str(item.get(key) or "") for key in ("query", "description"))
        fingerprint = f"{tool}|{'|'.join(ranges)}|{tuple(round(t, 3) for t in times)}|{' '.join(query_text.lower().split())}"
        records.append((fingerprint, tool, "|".join(ranges), times))
    unique: Dict[str, tuple[str, str, List[float]]] = {}
    for fingerprint, tool, range_text, times in records:
        unique.setdefault(fingerprint, (tool, range_text, times))
    if len(unique) < 2:
        # A single packed frame observation can still contain independent
        # timestamped views.  This is enough for simple visual checks; count
        # and negative/global contracts apply their stricter coverage rules.
        return bool(unique and max((len(times) for _, _, times in unique.values()), default=0) >= 2)
    timestamped = [times for _, _, times in unique.values()]
    if timestamped and all(timestamped) and len({t for times in timestamped for t in times}) < 2:
        return False
    tools_seen = {tool for tool, _range, _times in unique.values()}
    ranges_seen = {range_text for _tool, range_text, _times in unique.values() if range_text}
    if len(tools_seen) >= 2 or len(ranges_seen) >= 2:
        return True
    # Distinct timestamp sets from the same tool are independent only when
    # they are not exact replays of the same frame.
    timestamp_sets = {
        tuple(round(value, 3) for value in times)
        for _tool, _range, times in unique.values()
        if times
    }
    return len(timestamp_sets) >= 2


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _option_value_tokens(options: Mapping[str, str]) -> List[set[str]]:
    return [
        _content_tokens(text)
        for text in options.values()
        if str(text or "").strip()
    ]


def _tool_returned_unlisted_action_value(
    *,
    query: str,
    options: Mapping[str, str],
    steps: Iterable[Any],
) -> bool:
    """Detect a real tool value that cannot be mapped to the listed choices.

    This is deliberately limited to action-count questions. It prevents the
    planner from turning a tool's observed total (e.g. 6 visible objects) into
    a fabricated listed label (e.g. option C=5), while leaving ordinary prose
    answers to the model/ledger checks.
    """
    normalized_query = _semantic_query_text(query)
    if not any(marker in normalized_query for marker in _COUNT_QUERY_MARKERS):
        return False
    if not set(re.findall(r"[a-z]+", normalized_query)).intersection(_ACTION_QUERY_MARKERS):
        return False
    option_tokens = _option_value_tokens(options)
    option_numbers = {
        token
        for tokens in option_tokens
        for token in tokens
        if token.isdigit()
    }
    if not option_numbers:
        return False

    mismatch = False
    for item in _iter_native_observations(steps):
        if _canonical_source_tool(item.get("tool")) != "Video_Clip_QA":
            continue
        payload = _coerce_mapping(item.get("result"))
        tool_query = str(item.get("query") or payload.get("query") or "")
        if not tool_query or _semantic_query_text(tool_query) != normalized_query:
            continue
        answers, _ = _native_option_fields(item)
        if not answers:
            continue
        # Uncertainty or a corrected answer retracts an older numeric mismatch.
        observed_numbers = set()
        uncertain = False
        for answer in answers:
            uncertain |= bool(re.search(r"\b(?:unknown|unclear|uncertain|maybe|possibly)\b", answer, re.IGNORECASE))
            observed_numbers.update(token for token in _content_tokens(answer) if token.isdigit())
        mismatch = bool(observed_numbers and not uncertain and not observed_numbers.intersection(option_numbers))
    return mismatch


def _explicit_premise_mismatch(
    *,
    query: str,
    options: Mapping[str, str],
    verified_slots: Iterable[Mapping[str, Any]],
    steps: Iterable[Any] = (),
) -> bool:
    """Use only the latest native answer to this question and option set.

    Lexical non-overlap cannot establish a mismatch. Intermediate questions,
    uncertain observations and old answers cannot invalidate the task.
    """
    if not options:
        return False
    normalize = lambda text: " ".join(str(text).casefold().split())
    expected_options = {label: normalize(text) for label, text in options.items()}
    mismatch = False
    explicit = re.compile(
        r"(?:none\s+of\s+(?:the\s+)?(?:listed\s+)?options?\s+(?:match|apply|is|are)|"
        r"(?:answer|video|scene)\s+does\s+not\s+match\s+(?:the\s+)?(?:listed\s+)?options?|"
        r"(?:video|scene)\s+is\s+unrelated\s+to\s+(?:the\s+)?(?:question|premise)|"
        r"(?:question|premise)\s+is\s+(?:invalid|incorrect|false))",
        re.IGNORECASE,
    )
    for item in _iter_native_observations(steps):
        payload = _coerce_mapping(item.get("result"))
        tool_query = str(item.get("query") or payload.get("query") or "")
        tool_options = {label: normalize(text) for label, text in _OPTION_LINE_RE.findall(tool_query)}
        if _semantic_query_text(tool_query) != _semantic_query_text(query) or tool_options != expected_options:
            continue
        answers, _ = _native_option_fields(item)
        if not answers:
            continue
        # A later scoped answer replaces the earlier premise judgment,
        # including when the later answer reports uncertainty.
        mismatch = all(
            bool(explicit.search(answer))
            and not re.search(r"\b(?:maybe|possibly|uncertain|unclear|unsure)\b", answer, re.IGNORECASE)
            for answer in answers
        )
    return mismatch


def _requires_option_comparison(query: str, thresholds: Mapping[str, Any]) -> bool:
    """Infer comparison from the question, never dataset names or options."""
    if not _OPTION_LINE_RE.search(str(query or "")):
        return False
    if bool(thresholds.get("require_option_comparison_all_tasks")):
        return True
    semantic_query = _semantic_query_text(query)
    return bool(re.search(
        r"\b(?:compare|compared|comparison|versus|vs|alternative|different|difference|"
        r"deviation|impact|anomal(?:y|ous|ies)|abnormal|unusual|"
        r"what\s+changed|which\s+one\s+stands\s+out|"
        r"not|never|without|doesn't|didn't|absent)\b",
        semantic_query,
        re.IGNORECASE,
    ))


def _native_option_fields(item: Mapping[str, Any]) -> tuple[List[str], List[str]]:
    """Return native answer fields and all native explanatory text separately."""
    payload = _coerce_mapping(item.get("result"))
    parsed = _coerce_mapping(item.get("parsed_json"))
    if not parsed:
        parsed = _coerce_mapping(payload.get("parsed_json"))
    answers: List[str] = []
    texts: List[str] = []
    for source in (parsed, payload, item):
        for key in ("answer", "answer_value"):
            value = source.get(key) if isinstance(source, Mapping) else None
            if isinstance(value, Mapping):
                value = value.get("answer") or value.get("value") or value.get("text")
            if value not in (None, ""):
                answers.append(str(value))
        for key in (
            "answer", "answer_value", "evidence", "evidence_items", "video_summary",
            "video_summary_value", "summary", "result",
        ):
            value = source.get(key) if isinstance(source, Mapping) else None
            if isinstance(value, Mapping):
                value = value.get("answer") or value.get("value") or value.get("text")
            if isinstance(value, (list, tuple, set)):
                texts.extend(str(entry) for entry in value)
            elif value not in (None, ""):
                texts.append(str(value))
    return answers, texts


def _has_native_anomaly_comparison(
    *,
    answer_label: str,
    options: Mapping[str, str],
    steps: Iterable[Any],
    query: str,
) -> bool:
    """Validate anomaly classification from a native visual answer.

    Python/SQL narration and Observer slots cannot establish a visual anomaly.
    A successful native answer may either state the selected class together
    with ruled-out alternatives, or return the selected class as a concise
    categorical answer with an explicit supporting explanation.
    """
    if not re.search(r"\b(?:anomal(?:y|ous|ies)|abnormal|unusual)\b", _semantic_query_text(query)):
        return False
    selected_tokens = _content_tokens(options.get(answer_label, ""))
    selected_tokens -= {"anomaly", "anomal", "type", "kind"}
    if not selected_tokens:
        return False
    alternative_tokens: set[str] = set()
    for label, text in options.items():
        if label != answer_label:
            alternative_tokens.update(_content_tokens(text))
    negative_markers = {
        "no", "not", "without", "absent", "absence", "none", "unrelated",
        "incompatible", "exclude", "excluded", "ruled", "rather", "instead",
    }
    for item in _iter_native_observations(steps):
        answers, texts = _native_option_fields(item)
        all_text = " ".join(texts).lower()
        tokens = _content_tokens(all_text)
        positive = selected_tokens.intersection(tokens)
        if not positive:
            continue
        has_alternative_exclusion = bool(
            negative_markers.intersection(tokens)
            and alternative_tokens.intersection(tokens)
        )
        if has_alternative_exclusion:
            return True
        # A direct native categorical answer is still a valid relation when it
        # is exact, non-uncertain, and accompanied by a concrete explanation.
        # This covers a short ``Shoplifting`` answer followed by an evidence
        # sentence, while refusing a bare Observer/Python claim.
        for answer_text in answers:
            answer_tokens = _content_tokens(answer_text)
            if selected_tokens.issubset(answer_tokens) and not any(
                marker in answer_text.lower()
                for marker in ("unknown", "uncertain", "unclear", "maybe", "possibly")
            ) and len(tokens) >= len(selected_tokens) + 3:
                return True
    return False


def _has_native_negative_option_relation(
    *,
    answer_label: str,
    options: Mapping[str, str],
    steps: Iterable[Any],
    query: str,
) -> bool:
    """Recognize a native answer to a ``not/never/without`` option query."""
    semantic_query = _semantic_query_text(query)
    if not re.search(
        r"\b(?:not|never|without|does\s+not|doesn't|did\s+not|didn't|absent)\b",
        semantic_query,
        re.IGNORECASE,
    ):
        return False
    selected_tokens = _content_tokens(options.get(answer_label, ""))
    selected_tokens -= {"a", "an", "the"}
    if not selected_tokens:
        return False
    negation_patterns = (
        r"\b(?:did|does|do)\s+not\b",
        r"\b(?:didn|doesn|don)['’]?t\b",
        r"\bnever\b", r"\bwithout\b", r"\bnot\b", r"\babsent\b",
        r"\bno\b",
    )
    def _selected_token_is_negated(clause: str) -> bool:
        """Require negation in the same clause as the selected option."""
        clause_text = " ".join(str(clause or "").lower().split())
        if not selected_tokens.intersection(_content_tokens(clause_text)):
            return False
        return any(re.search(pattern, clause_text) for pattern in negation_patterns)

    for item in _iter_native_observations(steps):
        answers, texts = _native_option_fields(item)
        all_text = " ".join(texts).lower()
        clauses = [
            clause
            for clause in re.split(
                r"(?:[.!?;]|\bbut\b|\bhowever\b|\balthough\b|\bwhile\b|\bwhereas\b)",
                all_text,
            )
            if clause.strip()
        ]
        selected_negative = any(
            _selected_token_is_negated(clause)
            for clause in clauses
        )
        # A complete native answer may put the selected phrase and the
        # negation in one sentence; retain an explicit mismatch form for
        # compact JSON responses.
        if not selected_negative:
            selected_negative = bool(
                re.search(
                    r"(?:none\s+of\s+the\s+(?:listed\s+)?options|"
                    r"not\s+(?:shown|performed|present|visible))",
                    all_text,
                )
                and selected_tokens.intersection(_content_tokens(all_text))
            )
        if not selected_negative:
            continue
        # The native answer itself may be a complete negative response.  If
        # it explicitly selects the listed option, no synthetic alternative
        # slot is needed; the explanation remains required above.
        for answer_text in answers:
            answer_tokens = _content_tokens(answer_text)
            if selected_tokens.issubset(answer_tokens) and len(answer_tokens) >= 1:
                return True
        # The local clause already establishes the selected option's absence.
        return True

    # A complete native sequence can establish an absent alternative without
    # putting the word ``not`` next to it.  Require explicit endpoint/sequence
    # language and at least two *other* listed alternatives in the same native
    # corpus; an incomplete one-line action description must remain unresolved.
    sequence_texts = list(_native_answer_texts(steps))
    # Video_Summary is not a categorical option answer, but a successful
    # transcript/summary can enumerate a complete post-event sequence.  Use
    # it only for the guarded completeness branch below; it never establishes
    # a positive option by itself.
    for step in steps:
        observations = (
            step.get("observation_items") or []
            if isinstance(step, Mapping)
            else getattr(step, "observation_items", None) or []
        )
        for item in _iter_observation_items(observations):
            if _canonical_source_tool(item.get("tool")) != "Video_Summary":
                continue
            if str(item.get("status") or "").strip().lower() != "success":
                continue
            summary_text = _native_observation_text(item).strip()
            if summary_text:
                sequence_texts.append(summary_text)
    native_corpus = " ".join(sequence_texts).lower()
    completeness_markers = (
        "after the performance", "after performance", "at the end", "following the performance",
        "culminating", "the sequence", "concluded", "finally",
    )
    if any(marker in native_corpus for marker in completeness_markers):
        present_labels = set()
        for label, option_text in options.items():
            option_tokens = _content_tokens(option_text)
            if not option_tokens:
                continue
            if option_tokens.intersection(_content_tokens(native_corpus)):
                present_labels.add(label)
        selected_tokens = _content_tokens(options.get(answer_label, ""))
        # Completeness alone cannot distinguish two simultaneously absent
        # alternatives (for example, a summary mentioning the jump and hug
        # but omitting both a high-five and a bow).  Accept this inference only
        # when the exhaustive sequence leaves exactly one listed option absent.
        absent_labels = set(options) - present_labels
        if (
            len(present_labels - {answer_label}) >= 2
            and absent_labels == {answer_label}
            and not selected_tokens.intersection(_content_tokens(native_corpus))
        ):
            return True
    return False


def _has_native_comparison_relation(
    *,
    answer_label: str,
    options: Mapping[str, str],
    steps: Iterable[Any],
    query: str,
) -> bool:
    """Recognize a direct native answer that names the compared target."""
    semantic_query = _semantic_query_text(query)
    if not re.search(
        r"\b(?:different|difference|compare|comparison|versus|vs|other|deviation|impact)\b",
        semantic_query,
        re.IGNORECASE,
    ):
        return False
    selected_tokens = _content_tokens(options.get(answer_label, ""))
    selected_tokens -= {"a", "an", "the", "part", "person", "object"}
    if not selected_tokens:
        return False
    comparison_markers = {
        "different", "difference", "same", "similar", "only", "unique", "versus",
        "compared", "comparison", "other", "others", "unlike", "distinct",
    }
    generic_tokens = {
        "part", "person", "people", "object", "thing", "one", "same",
        "different", "other", "others",
    }

    def _specific_option_tokens(option_text: str) -> set[str]:
        return _content_tokens(option_text) - generic_tokens

    def _option_named(option_tokens: set[str], evidence_tokens: set[str]) -> bool:
        if not option_tokens:
            return False
        return all(
            token in evidence_tokens
            or bool(_CONTENT_TOKEN_EQUIVALENTS.get(token, frozenset()).intersection(evidence_tokens))
            for token in option_tokens
        )

    for item in _iter_native_observations(steps):
        answers, texts = _native_option_fields(item)
        all_text = " ".join(texts).lower()
        tokens = _content_tokens(all_text)
        if not comparison_markers.intersection(tokens):
            continue

        # An explicitly named native answer is stronger than a generic
        # yes/no statement. Keep an explanation requirement so a bare
        # categorical answer cannot bootstrap a comparison.
        selected_specific = _specific_option_tokens(options.get(answer_label, ""))
        for answer_text in answers:
            answer_tokens = _content_tokens(answer_text)
            if (
                selected_specific
                and selected_specific.issubset(answer_tokens)
                and len(tokens) >= len(selected_specific) + 3
                and not any(
                    marker in answer_text.lower()
                    for marker in ("unknown", "uncertain", "unclear", "maybe", "possibly")
                )
            ):
                return True

        # If the native answer is not labelled, infer a choice only when
        # exactly one listed option is named by the evidence. If the
        # explanation mentions both feet and hands, it is unresolved
        # multi-target evidence and must not satisfy either option.
        named_options: set[str] = set()
        for label, option_text in options.items():
            option_specific = _specific_option_tokens(option_text)
            if not option_specific:
                continue
            if _option_named(option_specific, tokens):
                named_options.add(label)
        if len(named_options) == 1 and answer_label in named_options:
            return True
    return False


def _comparison_option_is_ambiguous(
    *,
    answer_label: str,
    options: Mapping[str, str],
    verified_slots: Iterable[Mapping[str, Any]],
    query: str,
    steps: Optional[Iterable[Any]] = None,
) -> bool:
    """Reject a comparison answer when one direct claim supports two options.

    A slot such as ``bare feet and hands`` is evidence that both listed
    attributes differ.  It cannot safely select either ``Foot`` or ``Hand``
    in a single-choice question unless another direct claim explicitly rules
    one of them out.  Numeric max/min comparisons are handled by their own
    count gate and are excluded here.
    """
    if not options or not answer_label or answer_label not in options:
        return False
    semantic_query = _semantic_query_text(query)
    if re.search(
        r"\b(?:not|never|without|does\s+not|doesn't|did\s+not|didn't|absent)\b",
        semantic_query,
        re.IGNORECASE,
    ):
        return False
    if _COMPARATIVE_COUNT_RE.search(semantic_query) or re.search(
        r"\b(?:how many|number of|count|total)\b", semantic_query, re.IGNORECASE
    ):
        return False
    if not re.search(
        r"\b(?:different|difference|compare|comparison|versus|vs|other|deviation|impact)\b",
        semantic_query,
        re.IGNORECASE,
    ):
        return False

    option_tokens = {
        label: _content_tokens(text) - {"part", "person", "people", "object", "thing"}
        for label, text in options.items()
    }
    positive_markers = {
        "different", "difference", "unlike", "distinct", "contrast", "versus",
        "compared", "comparison", "bare", "boots", "bandaged", "covered",
        "uncovered", "unique", "only",
    }
    negative_markers = {
        "same", "similar", "normal", "identical", "unchanged", "not", "no",
        "without", "absent", "none",
    }
    positive_labels: set[str] = set()
    negative_labels: set[str] = set()
    multi_positive_witnesses: List[set[str]] = []
    for item in verified_slots:
        if not isinstance(item, Mapping) or not _slot_has_direct_visual_source(item, steps):
            continue
        claim_text = " ".join(
            str(item.get(key, "") or "")
            for key in (
                "label", "target", "relation", "direction", "observed_value", "value",
            )
        )
        claim_tokens = _content_tokens(claim_text)
        relation = str(item.get("relation") or "").strip().lower()
        is_positive = bool(
            positive_markers.intersection(claim_tokens)
            or relation in {"difference", "different", "different_from", "contrast", "unlike"}
        )
        is_negative = bool(
            negative_markers.intersection(claim_tokens)
            or relation in {"same", "similar", "equal", "similarity", "not_present", "absence", "absent"}
        )
        if not is_positive and not is_negative:
            continue
        matched: set[str] = set()
        for label, tokens in option_tokens.items():
            if not tokens:
                continue
            if any(
                token in claim_tokens
                or bool(_CONTENT_TOKEN_EQUIVALENTS.get(token, frozenset()).intersection(claim_tokens))
                for token in tokens
            ):
                matched.add(label)
        if is_positive:
            positive_labels.update(matched)
            if len(matched) >= 2 and answer_label in matched:
                multi_positive_witnesses.append(matched)
        if is_negative:
            negative_labels.update(matched)

    supported = positive_labels.intersection(options)
    if answer_label in supported and len(supported) >= 2:
        competitors = supported - {answer_label}
        if not competitors.issubset(negative_labels):
            return True
    for witness in multi_positive_witnesses:
        competitors = witness - {answer_label}
        if competitors and not competitors.issubset(negative_labels):
            return True
    return False


def _has_option_comparison_evidence(
    *,
    answer_label: str,
    options: Mapping[str, str],
    verified_slots: Iterable[Mapping[str, Any]],
    steps: Iterable[Any] = (),
    query: str = "",
) -> bool:
    if not options or not answer_label or answer_label not in options:
        return False
    if _has_native_anomaly_comparison(
        answer_label=answer_label,
        options=options,
        steps=steps,
        query=query,
    ):
        return True
    if _has_native_negative_option_relation(
        answer_label=answer_label,
        options=options,
        steps=steps,
        query=query,
    ):
        return True
    if _has_native_comparison_relation(
        answer_label=answer_label,
        options=options,
        steps=steps,
        query=query,
    ):
        return True
    selected_tokens = _content_tokens(f"{answer_label} {options.get(answer_label, '')}")
    other_option_tokens = {
        label: _content_tokens(f"{label} {text}")
        for label, text in options.items()
        if label != answer_label
    }
    verified_items = [item for item in verified_slots if isinstance(item, Mapping)]
    # Numeric max/min claims are a comparison even when the selected option
    # text does not contain a lexical comparison marker (e.g. ``Apples``).
    # They are still protected by the count-redundancy contract for visual
    # counting, so this branch only recognizes the relation shape here.
    numeric_count_items = [
        item for item in verified_items
        if str(item.get("scope") or "").strip().lower()
        in {"action_count", "total_count", "state_count", "scene_count", "generic_count"}
        and re.search(r"\b\d+(?:\.\d+)?\b", str(item.get("observed_value") or item.get("value") or ""))
    ]
    explicit_numeric_comparison = any(
        str(item.get("scope") or "").strip().lower() == "comparison"
        and str(item.get("relation") or "").strip().lower() in {
            "max", "min", "largest", "smallest", "highest", "lowest",
        }
        and str(item.get("observed_value") or item.get("value") or "").strip()
        for item in verified_items
    )
    if len(numeric_count_items) >= 2 and explicit_numeric_comparison:
        selected_value = " ".join(
            str(item.get("observed_value") or item.get("value") or "")
            for item in verified_items
            if str(item.get("scope") or "").strip().lower() == "comparison"
        )
        if selected_tokens.intersection(_content_tokens(selected_value)):
            return True
    selected_comparison_anchor = False
    selected_direct_visual_anchor = False
    selected_claim_values: List[set[str]] = []
    ruled_out_alternatives: set[str] = set()
    # A valid comparison is often represented by several evidence slots: one
    # slot establishes the selected difference ("barefoot while others wear
    # shoes"), while later slots rule out the listed alternatives ("legs are
    # covered like the others", "eyes are normal", ...).  Requiring every
    # option token to occur in the *same* slot made a truthful multi-step
    # comparison fail its own safety gate and caused needless 15-turn loops.
    alternative_relation_markers = _COMPARISON_MARKERS.union({
        "same", "similar", "normal", "identical", "covered", "wearing", "only",
    })
    for item in verified_items:
        claim_text = " ".join(
            str(item.get(key, "") or "")
            for key in (
                "label", "target", "relation", "direction", "observed_value", "value",
            )
        )
        claim_tokens = _content_tokens(claim_text)
        relation = str(item.get("relation") or "").strip().lower()
        direction = str(item.get("direction") or "").strip().lower()
        negative_relation = bool(
            relation in {"absence", "absent", "not_present", "negative", "not_shown", "excluded"}
            or any(marker in direction for marker in ("not", "never", "absent", "without", "no"))
            or _NEGATION_MARKERS.intersection(claim_tokens)
        )
        has_count_order = bool(
            _COUNT_COMPARISON_MARKERS.intersection(claim_tokens)
            or relation in {"max", "min", "largest", "smallest", "highest", "lowest"}
            or ">" in direction
            or "<" in direction
        )
        selected_overlap = bool(
            selected_tokens.intersection(claim_tokens)
            or any(
                _CONTENT_TOKEN_EQUIVALENTS.get(token, frozenset()).intersection(claim_tokens)
                for token in selected_tokens
            )
        )
        if selected_overlap and (
            _COMPARISON_MARKERS.intersection(claim_tokens)
            or {"only", "unique"}.intersection(claim_tokens)
            or has_count_order
            or negative_relation
        ):
            selected_comparison_anchor = True
            selected_direct_visual_anchor = selected_direct_visual_anchor or _slot_has_direct_visual_source(item, steps)
        if selected_overlap and _slot_has_direct_visual_source(item, steps):
            selected_claim_values.append(
                _content_tokens(
                    " ".join(
                        str(item.get(key, "") or "")
                        for key in ("observed_value", "value", "direction")
                    )
                )
            )
        for label, tokens in other_option_tokens.items():
            if not tokens.intersection(claim_tokens):
                continue
            if alternative_relation_markers.intersection(claim_tokens):
                ruled_out_alternatives.add(label)
    if not selected_comparison_anchor and len(selected_claim_values) >= 2:
        non_empty_values = [value for value in selected_claim_values if value]
        if len({tuple(sorted(value)) for value in non_empty_values}) >= 2:
            # Two independently cited values for the selected attribute (for
            # example ``boots`` versus ``bare`` feet) are themselves a direct
            # comparison, even when the Observer omitted the word
            # ``different`` from the slot label.
            selected_comparison_anchor = True
            selected_direct_visual_anchor = True
    # A directly grounded relation is enough when the evidence itself states
    # the discriminative predicate (e.g. ``the turtle carried the jellyfish``
    # for a transition question). Requiring a negative statement for every
    # other listed option turned a decisive native Video_Clip_QA result into an
    # artificial recovery loop. Explicitly negated selected options are still
    # rejected by the independent safety gate.
    if selected_comparison_anchor:
        explicit_comparison_query = bool(
            re.search(
                r"\b(?:different|difference|compare|comparison|versus|vs|alternative|deviation|impact|overall)\b",
                _semantic_query_text(query),
                re.IGNORECASE,
            )
        )
        if explicit_comparison_query and not selected_direct_visual_anchor:
            return False
        if re.search(
            r"\b(?:not|never|without|does\s+not|doesn't|did\s+not|didn't|absent)\b",
            _semantic_query_text(query),
            re.IGNORECASE,
        ):
            # For a negative question, an anchor must also name a plausible
            # alternative or a native answer must have made the exclusion
            # explicit.  A slot containing only ``bow absent`` is not enough
            # to distinguish a missed observation from the requested answer.
            if ruled_out_alternatives:
                return True
            # A structured negative relation with a directly cited native
            # answer is safe even when the alternatives are not repeated in
            # the ledger; the native helper above handles that case.
            return False
        return True
    normalized_query = _semantic_query_text(query)
    direct_relation_markers = (
        "carried", "carry", "leave", "leaving", "escape", "rescued",
        "helped", "took", "brought", "before and after", "how did",
        "how does", "transition",
    )
    selected_relation_tokens = selected_tokens - {answer_label.lower()}
    if selected_relation_tokens and any(
        marker in normalized_query for marker in direct_relation_markers
    ):
        for step in steps:
            if isinstance(step, Mapping):
                observation_items = step.get("observation_items") or []
            else:
                observation_items = getattr(step, "observation_items", None) or []
            for item in _iter_observation_items(observation_items):
                if _canonical_source_tool(item.get("tool")) not in {
                    "Video_Clip_QA", "Action_Recognition", "Cross_Modal_Verify"
                }:
                    continue
                if str(item.get("status") or "").lower() != "success":
                    continue
                result_tokens = _content_tokens(item.get("result"))
                if selected_relation_tokens.issubset(result_tokens) or (
                    selected_relation_tokens.intersection(result_tokens)
                    and len(selected_relation_tokens.intersection(result_tokens)) >= max(1, min(2, len(selected_relation_tokens)))
                ):
                    return True
    # Preserve the original compact form for a single claim that names both
    # the selected option and an alternative explicitly.
    return any(
        selected_tokens.intersection(_content_tokens(f"{item.get('label', '')} {item.get('value', '')}"))
        and (
            _COMPARISON_MARKERS.intersection(
                _content_tokens(f"{item.get('label', '')} {item.get('value', '')}")
            )
            or _COUNT_COMPARISON_MARKERS.intersection(
                _content_tokens(f"{item.get('label', '')} {item.get('value', '')}")
            )
        )
        and any(
            tokens.intersection(_content_tokens(f"{item.get('label', '')} {item.get('value', '')}"))
            for tokens in other_option_tokens.values()
        )
        for item in verified_items
    )


def _claim_contract_text(item: Mapping[str, Any]) -> str:
    """Serialize the shared claim fields for conservative contract checks."""
    values: List[str] = []
    for key in (
        "claim_id", "scope", "label", "target", "relation", "direction",
        "qualifiers", "observed_value", "value", "citation", "source_tool",
        "time_window",
    ):
        value = item.get(key)
        if value not in (None, "", [], {}):
            values.append(str(value))
    return " ".join(values).lower()


def _has_redundant_count_evidence(
    *,
    verified_slots: Iterable[Mapping[str, Any]],
    steps: Iterable[Any] = (),
    query: str = "",
    answer_label: str = "",
    options: Optional[Mapping[str, str]] = None,
) -> bool:
    """Require independent, temporally adequate evidence for visual counts.

    A single Inspect_Frame pass can expose the right objects while the VLM
    still miscounts one category.  Reflector is text-only, so it cannot
    independently correct that visual error.  Two distinct cited timestamps
    or two evidence tools provide the minimum handoff boundary.  Python/SQL
    narration is never an independent visual source.  A zero answer also
    needs coverage of the complete requested video span; a few negative
    windows cannot establish that an event never occurs elsewhere.
    """
    steps = list(steps)
    count_items = [
        item for item in verified_slots
        if isinstance(item, Mapping)
        and str(item.get("scope") or "").strip().lower() in {
            "action_count", "total_count", "state_count", "scene_count",
            "generic_count", "comparison",
        }
        and not bool(item.get("context_only", False))
        and not bool(item.get("superseded", False))
        and _slot_has_direct_visual_source(item, steps)
    ]
    count_claims = [
        item for item in count_items
        if str(item.get("scope") or "").strip().lower()
        in {"action_count", "total_count", "state_count", "scene_count", "generic_count"}
    ]
    # A relative largest/smallest answer may have only one structured
    # comparison slot and no exact per-category numbers.  That slot still
    # needs independent visual support; it must not be rejected merely
    # because the question did not ask for numeric totals.
    if not count_claims:
        count_claims = [
            item for item in count_items
            if str(item.get("scope") or "").strip().lower() == "comparison"
        ]
    if not count_claims:
        return False

    def _field_values(item: Mapping[str, Any], *keys: str) -> List[str]:
        values: List[str] = []
        for key in keys:
            raw = item.get(key)
            raw_values = raw if isinstance(raw, (list, tuple, set)) else [raw]
            for value in raw_values:
                text = str(value or "").strip()
                if text and text not in values:
                    values.append(text)
        return values

    def _sources(item: Mapping[str, Any]) -> set[str]:
        return {
            _canonical_source_tool(value)
            for value in _field_values(item, "source_tool", "supporting_source_tools")
            if str(value or "").strip()
        }

    def _timestamps(item: Mapping[str, Any]) -> set[str]:
        citation_text = " ".join(
            _field_values(
                item,
                "citation",
                "time_window",
                "supporting_citations",
                "supporting_time_windows",
            )
        )
        return set(
            re.findall(
                r"\b\d+(?:\.\d+)?\s*(?:s|sec|secs|seconds)\b",
                citation_text,
                re.IGNORECASE,
            )
        )

    def _range_from_value(value: Any) -> Optional[tuple[float, float]]:
        if isinstance(value, Mapping):
            for start_key, end_key in (
                ("start", "end"), ("start_time", "end_time"),
                ("t_start", "t_end"),
            ):
                if start_key in value and end_key in value:
                    try:
                        start = float(value[start_key])
                        end = float(value[end_key])
                    except (TypeError, ValueError):
                        continue
                    if end >= start:
                        return (start, end)
            return None
        text = str(value or "")
        match = re.search(
            r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?\s*[-–]\s*"
            r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        try:
            start, end = float(match.group(1)), float(match.group(2))
        except (TypeError, ValueError):
            return None
        return (start, end) if end >= start else None

    def _observation_range(observation: Mapping[str, Any]) -> Optional[tuple[float, float]]:
        for key in (
            "effective_time_range", "requested_time_range", "time_range",
            "timestamp_range", "search_window",
        ):
            value = observation.get(key)
            parsed = _range_from_value(value)
            if parsed is not None:
                return parsed
        for start_key, end_key in (("t_start", "t_end"), ("start_time", "end_time")):
            if observation.get(start_key) is not None and observation.get(end_key) is not None:
                parsed = _range_from_value({"start": observation[start_key], "end": observation[end_key]})
                if parsed is not None:
                    return parsed
        raw_times = observation.get("image_timestamps") or observation.get("sample_times") or []
        if isinstance(raw_times, (list, tuple, set)):
            numeric: List[float] = []
            for value in raw_times:
                try:
                    numeric.append(float(value))
                except (TypeError, ValueError):
                    continue
            if numeric:
                return (min(numeric), max(numeric))
        return None

    def _merge_ranges(ranges: List[tuple[float, float]]) -> List[tuple[float, float]]:
        merged_ranges: List[tuple[float, float]] = []
        for start, end in sorted(ranges):
            if not merged_ranges or start > merged_ranges[-1][1] + 1e-6:
                merged_ranges.append((start, end))
            else:
                merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        return merged_ranges

    visual_ranges: List[tuple[float, float]] = []
    native_ranges: List[tuple[float, float]] = []
    visual_timestamps: set[str] = set()
    returned_visual_tools: set[str] = set()
    source_durations: List[float] = []
    for step in steps:
        observation_items = (
            step.get("observation_items") or []
            if isinstance(step, Mapping)
            else getattr(step, "observation_items", None) or []
        )
        for observation in _iter_observation_items(observation_items):
            tool = _canonical_source_tool(observation.get("tool"))
            status = str(observation.get("status") or "").strip().lower()
            if status in {"error", "no_valid_result", "failed"}:
                continue
            if status == 'success':
                for metadata in (observation, _coerce_mapping(observation.get('result'))):
                    value = metadata.get('source_video_duration_seconds')
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
                        source_durations.append(float(value))
            has_images = bool(_observation_image_assets(observation))
            if has_images:
                returned_visual_tools.add(tool)
            observed_range = _observation_range(observation)
            if observed_range is not None:
                if tool in {"Inspect_Frame", "Inspect_Clip", "Crop", "Zoom_in"} or has_images:
                    visual_ranges.append(observed_range)
                if tool in {"Video_Clip_QA", "Action_Recognition", "Cross_Modal_Verify"}:
                    native_ranges.append(observed_range)
            if tool in {"Inspect_Frame", "Inspect_Clip", "Crop", "Zoom_in"} or has_images:
                raw_times = observation.get("image_timestamps") or observation.get("sample_times") or []
                if isinstance(raw_times, (list, tuple, set)):
                    for value in raw_times:
                        try:
                            visual_timestamps.add(f"{float(value):.3f}")
                        except (TypeError, ValueError):
                            continue
                for image in observation.get("image_evidence") or []:
                    if not isinstance(image, Mapping):
                        continue
                    for value in image.get("sample_times") or []:
                        try:
                            visual_timestamps.add(f"{float(value):.3f}")
                        except (TypeError, ValueError):
                            continue

    merged_visual = _merge_ranges(visual_ranges)
    merged_native = _merge_ranges(native_ranges)
    visual_view_count = len(visual_timestamps)
    valid_visual_sources = {
        "Video_Clip_QA", "Action_Recognition", "Cross_Modal_Verify", "Inspect_Frame",
        "Inspect_Clip", "Crop", "Zoom_in",
    } | returned_visual_tools

    # A claim must cite a real visual/native source.  ``Python_Executor`` is
    # deliberately excluded even when its prose repeats a frame description.
    for item in count_claims:
        sources = _sources(item).union({
            _canonical_source_tool(source)
            for source in _field_values(item, "citation", "supporting_citations")
            if _canonical_source_tool(source) in valid_visual_sources
        })
        if not sources.intersection(valid_visual_sources):
            return False

    # At least two independently timestamped views/tools are required.  A
    # native count may be one of those views, but a single native response is
    # not allowed to stand alone because it can inherit the same VLM error as
    # the Planner.
    cited_timestamp_count = max(
        (len(_timestamps(item)) for item in count_claims),
        default=0,
    )
    distinct_claim_sources = set().union(*(_sources(item) for item in count_claims))
    independent_views = (
        cited_timestamp_count >= 2
        or visual_view_count >= 2
        or len(merged_native) >= 2
        or len(merged_visual) >= 2
        or (
            len(distinct_claim_sources.intersection(valid_visual_sources)) >= 2
            and (len(merged_native) >= 2 or len(merged_visual) >= 2)
        )
    )
    if not independent_views:
        return False

    normalized_query = _semantic_query_text(query)
    selected_text = str(options.get(answer_label, "") if isinstance(options, Mapping) else "")
    answer_text = selected_text or str(answer_label or "")
    if not answer_text:
        answer_text = normalized_query
    answer_tokens = _content_tokens(answer_text)
    zero_answer = bool(answer_tokens.intersection({"0", "zero", "none", "no"}))
    # Exact/global counts need temporal coverage, not just two independent
    # snapshots.  A local query (for example ``at 00:29`` or ``at the
    # beginning``) is exempt because its scope is already bounded by the
    # question.  The same gate applies to positive and zero global counts;
    # otherwise a sparse scan can silently promote a missed event to a total.
    local_scope_query = bool(re.search(
        r"\b(?:at|around|near|between|from)\s+(?:\d|\d\s*[:.]|the\s+)?"
        r"(?:beginning|start|end|first|last|\d)",
        normalized_query,
    )) or bool(re.search(
        r"\b(?:at|in|near)\s+the\s+(?:beginning|start|end|first|last)\b|"
        r"\b(?:first|last)\s+(?:time|scene|occurrence|action|event)\b|"
        r"\bin\s+(?:this|the)\s+(?:short\s+)?window\b",
        normalized_query,
    ))
    global_scope_query = (
        not local_scope_query
        and any(marker in normalized_query for marker in (
            "throughout this video", "throughout the video", "throughout video",
            "in this video", "in the video", "within this video", "within the video",
            "in total", "total count", "total number", "overall",
        ))
    )
    if global_scope_query and any(marker in normalized_query for marker in _COUNT_QUERY_MARKERS):
        # Only executable media metadata can establish the full source extent.
        # The last observed/requested timestamp is not the video duration.
        if not source_durations:
            return False
        source_duration = max(source_durations)
        # Treat a full-span observation as explicit only when it starts near
        # zero, covers a substantial or complete clip, and has bounded gaps.
        # This blocks the historical failure where a few short windows were
        # promoted to a global count.
        coverage_ranges = _merge_ranges(visual_ranges + native_ranges)
        if not coverage_ranges:
            return False
        first_start = coverage_ranges[0][0]
        last_end = coverage_ranges[-1][1]
        union_length = sum(end - start for start, end in coverage_ranges)
        span = max(0.0, last_end - first_start)
        max_gap = max((coverage_ranges[idx + 1][0] - coverage_ranges[idx][1]
                       for idx in range(len(coverage_ranges) - 1)), default=0.0)
        full_span = (
            first_start <= 1.0
            and last_end >= source_duration - min(1.0, source_duration * 0.01)
            and span >= min(5.0, source_duration * 0.80)
            and union_length / source_duration >= 0.80
            and max_gap <= max(8.0, 0.15 * span)
        )
        # Requested t_start/t_end ranges can look complete while the actual
        # frame sampler only inspected a handful of points. A global zero
        # claim needs endpoint coverage and bounded gaps between sampled
        # frames, otherwise a short event between sparse samples is silently
        # promoted to zero.
        sampled_times = sorted(
            float(value)
            for value in visual_timestamps
            if value not in (None, "")
        )
        if sampled_times:
            sampled_span = sampled_times[-1] - sampled_times[0]
            sampled_gap = max(
                (
                    sampled_times[idx + 1] - sampled_times[idx]
                    for idx in range(len(sampled_times) - 1)
                ),
                default=float("inf"),
            )
            full_span = full_span and (
                len(sampled_times) >= 4
                and sampled_times[0] <= first_start + 1.0
                and sampled_times[-1] >= last_end - 2.0
                and sampled_gap <= max(10.0, 0.05 * max(sampled_span, 1.0))
            )
        else:
            full_span = False
        if not full_span:
            return False

    return True


def _claim_contract_requirements(query: str, thresholds: Mapping[str, Any]) -> List[str]:
    """Infer only semantic obligations explicitly present in the query."""
    # Mix prompts append the answer options and an instruction such as
    # ``Reply with exactly one ...``.  Those protocol words are not question
    # semantics: treating ``exactly`` as a qualifier made ordinary
    # EgoSchema activity questions impossible to finish.  Scope this contract
    # to the natural-language question line/paragraph and ignore the option
    # rendering and output-format suffix.
    raw_query = str(query or "")
    semantic_query = _semantic_query_text(raw_query)
    normalized = semantic_query
    requirements: List[str] = []
    has_count_semantics = bool(
        re.search(
            r"\b(?:how\s+many|number\s+of|count(?:ing\s+problem)?|action\s+count|moving\s+count)\b",
            normalized,
        )
        or _COMPARATIVE_COUNT_RE.search(normalized)
    )
    action_count_query = _query_asks_action_count(semantic_query)
    comparative_count_query = bool(_COMPARATIVE_COUNT_RE.search(normalized))
    if action_count_query:
        requirements.append("action_count")
    elif has_count_semantics and not comparative_count_query:
        # ``Counting Problem`` is a dataset task label, not evidence that the
        # protagonist performed an action.  Static scene/category counts must
        # not be forced through the action-count contract.
        requirements.append("scene_count")
    if comparative_count_query:
        requirements.append("comparison")
    if has_count_semantics:
        requirements.append("count_redundancy")
    # A contextual lead-in is not a request to reconstruct an event order.
    # Keep the remainder intact so an actual sequence question still gates.
    order_query = re.sub(
        r"^(?:considering|from|based on|given)\s+the\s+sequence\s+of\s+events\s*,\s*",
        "", " ".join(semantic_query.lower().split()),
    )
    if (
        re.search(r"\b(?:order|sequence|chronological)\b", order_query)
    ):
        requirements.append("order")
    if _has_dynamic_relation(normalized):
        requirements.append("dynamic_relation")
    if _requires_option_comparison(raw_query, thresholds) or any(
        marker in normalized
        for marker in ("compare", "comparison", "different from", "different than", "versus", " vs ", "alternative")
    ):
        requirements.append("comparison")
    if re.search(
        r"\b(?:depending|differently|various\s+types?|specific\s+task|"
        r"respective(?:ly)?|primary\s+objective|main\s+methods?)\b",
        semantic_query,
        re.IGNORECASE,
    ):
        requirements.append("comparison")
    # Static attributes can be established by one clear, grounded observation.
    # A color/body-part keyword alone does not imply independent-view evidence.
    # Identity, rationale and interaction questions retain their additional gate;
    # counts, comparisons and temporal obligations are handled separately.
    if re.search(
        r"\b(?:what\s+does\b.*\b(?:with|using|from|off)\b|who|why)\b",
        semantic_query,
        re.IGNORECASE,
    ):
        requirements.append("answer_redundancy")
    qualifier_markers = (
        "exactly", "at least", "at most", "only", "first", "last", "immediately",
        "before", "after", "left", "right", "directly", "without", "never",
    )
    if any(
        re.search(rf"\b{re.escape(marker)}\b", normalized)
        for marker in qualifier_markers
    ):
        requirements.append("qualifier")
    if _query_asks_purpose(semantic_query) or _query_asks_rationale(semantic_query):
        requirements.append("purpose")
    if _query_requires_independent_evidence(raw_query, thresholds):
        requirements.append("evidence_redundancy")
    return list(dict.fromkeys(requirements))


def _audit_claim_contract(
    *,
    query: str,
    verified_slots: Iterable[Mapping[str, Any]],
    steps: Iterable[Any],
    answer_label: str,
    options: Mapping[str, str],
    thresholds: Mapping[str, Any],
) -> Dict[str, Any]:
    """Audit Planner/Observer claims jointly before accepting a Finish.

    The contract intentionally reports gaps instead of inventing evidence.
    Native relation/count/order results may satisfy a dimension only when the
    tool itself returned a successful, cited fact or the ledger preserved the
    corresponding structured field.
    """
    required = _claim_contract_requirements(query, thresholds)
    items = [
        item for item in verified_slots
        if isinstance(item, Mapping)
        and not bool(item.get("context_only", False))
        and not bool(item.get("superseded", False))
        and not _slot_is_synthetic_only(item)
        and not bool(item.get("logical conflict", item.get("logical_conflict", False)))
    ]
    texts = [_claim_contract_text(item) for item in items]
    satisfied: Dict[str, bool] = {key: False for key in required}

    if "action_count" in satisfied:
        satisfied["action_count"] = any(
            str(item.get("scope") or "").strip().lower() == "action_count"
            or bool(_DIRECT_ACTION_COUNT_RE.search(text))
            for item, text in zip(items, texts)
        )
    if "scene_count" in satisfied:
        satisfied["scene_count"] = any(
            str(item.get("scope") or "").strip().lower()
            in {"total_count", "state_count", "scene_count", "generic_count"}
            and str(item.get("scope") or "").strip().lower() != "action_count"
            for item in items
        )
    if "order" in satisfied:
        observed_order = _observed_order_sequence(query=query, verified_slots=items)
        satisfied["order"] = bool(observed_order) or any(
            any(marker in text for marker in ("order", "sequence", "chronological", "->"))
            and len(re.findall(r"\b(?:event|step)?\s*\d+\b", text)) >= 2
            for text in texts
        ) or _has_order_answer_support(
            answer_label=answer_label,
            options=options,
            query=query,
            verified_slots=items,
            steps=steps,
        )
    if "dynamic_relation" in satisfied:
        direct_native_relation = False
        for step in steps:
            if isinstance(step, Mapping):
                observation_items = step.get("observation_items") or []
            else:
                observation_items = getattr(step, "observation_items", None) or []
            for observation in _iter_observation_items(observation_items):
                tool = _canonical_source_tool(observation.get("tool"))
                status = str(observation.get("status") or "").strip().lower()
                if tool not in {"Video_Clip_QA", "Action_Recognition", "Cross_Modal_Verify"}:
                    continue
                if status != "success":
                    continue
                result_text = str(observation.get("result") or "").lower()
                if _has_dynamic_relation(result_text):
                    direct_native_relation = True
                    break
            if direct_native_relation:
                break
        satisfied["dynamic_relation"] = direct_native_relation or any(
            bool(item.get("relation") or item.get("direction") or item.get("target"))
            or _has_dynamic_relation(text)
            for item, text in zip(items, texts)
        )
    if "comparison" in satisfied:
        satisfied["comparison"] = _has_option_comparison_evidence(
            answer_label=answer_label,
            options=options,
            verified_slots=items,
            steps=steps,
            query=query,
        ) or any(
            bool(item.get("relation") or item.get("direction"))
            and any(
                marker in text
                for marker in (
                    "different", "same", "only", "versus", "compared",
                    "largest", "smallest", "highest", "lowest", "greatest",
                    "fewest", "most", "least", "more", "fewer", "greater",
                    "maximum", "minimum", "max", "min", ">",
                )
            )
            for item, text in zip(items, texts)
        )
    if "purpose" in satisfied:
        slot_rationale = any(
            any(
                marker in text
                for marker in (
                    "purpose", "function", "used for", "used to", "serves to",
                    "to clean", "to rinse", "to store", "to dispose", "to provide",
                    "because", "ensure", "ensures", "so that", "in order to",
                    "helps", "allows", "reason is", "goal", "objective",
                )
            )
            and bool(item.get("relation") or item.get("direction") or item.get("value"))
            for item, text in zip(items, texts)
        )
        satisfied["purpose"] = slot_rationale or _has_native_rationale_support(steps)
    if "qualifier" in satisfied:
        query_qualifiers = [
            marker for marker in (
                "exactly", "at least", "at most", "only", "first", "last", "immediately",
                "before", "after", "left", "right", "directly", "without", "never",
            )
            if re.search(
                rf"\b{re.escape(marker)}\b",
                _semantic_query_text(query),
                re.IGNORECASE,
            )
        ]
        satisfied["qualifier"] = bool(query_qualifiers) and _has_query_qualifier_support(
            query=query,
            verified_slots=items,
            steps=steps,
        )
    if "count_redundancy" in satisfied:
        satisfied["count_redundancy"] = _has_redundant_count_evidence(
            verified_slots=items,
            steps=steps,
            query=query,
            answer_label=answer_label,
            options=options,
        )
    if "evidence_redundancy" in satisfied:
        satisfied["evidence_redundancy"] = _has_independent_evidence(steps=steps)
    if "answer_redundancy" in satisfied:
        satisfied["answer_redundancy"] = _has_semantic_answer_redundancy(
            answer_label=answer_label,
            options=options,
            query=query,
            verified_slots=items,
            steps=steps,
        )

    missing = [key for key in required if not satisfied.get(key, False)]
    return {
        "status": "pass" if not missing else "blocked",
        "required": required,
        "satisfied": satisfied,
        "missing": missing,
        "verified_claim_count": len(items),
        "contract_version": 1,
    }


def _selected_option_is_negated(
    *,
    answer_label: str,
    options: Mapping[str, str],
    verified_slots: Iterable[Mapping[str, Any]],
) -> bool:
    if not options or not answer_label or answer_label not in options:
        return False
    selected_tokens = _content_tokens(options.get(answer_label, ""))
    if not selected_tokens:
        return False
    for item in verified_slots:
        # Scope negation to the clause naming the option. A negative claim
        # about a different event must not negate a positive selected event.
        for clause in re.split(r"[.;\n]|\bbut\b|\bwhereas\b", str(item.get("value") or ""), flags=re.IGNORECASE):
            tokens = _content_tokens(clause)
            if selected_tokens.issubset(tokens) and _NEGATION_MARKERS.intersection(tokens):
                return True
    return False


class EvidenceVerifier:
    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        payload = dict(config or load_evolution_config())
        self.thresholds = dict(payload.get("verification") or {})

    def verify_answerability(
        self,
        *,
        evidence_slots: List[Dict[str, Any]],
        steps: Iterable[Any],
        answer: str,
        completion_basis: str,
        query: str = "",
    ) -> VerificationResult:
        step_list = list(steps)
        # Superseded ledger entries remain in raw trajectory/history for
        # auditability, but must not participate in the current safety score.
        # Otherwise an old ``unknown`` action slot can keep blocking a later
        # direct visual observation forever.
        slots = [
            item for item in evidence_slots
            if isinstance(item, dict) and not bool(item.get("superseded", False))
        ]
        # Retain the full verified ledger for conflict and contract checks;
        # answer-token relevance must not hide contrary observations.
        all_verified_slots = [
            item for item in slots
            if str(item.get("evidence_status") or "").lower() == "verified"
            and not _slot_is_uncertain(item)
            and _slot_has_executed_source(item, step_list)
        ]
        options = {label: text for label, text in _OPTION_LINE_RE.findall(str(query or ""))}
        answer = normalize_answer_format(answer, options)
        answer_label_match = re.search(r"\(([A-Z])\)", str(answer or "").upper())
        answer_label = answer_label_match.group(1) if answer_label_match else ""
        negative_option_query = bool(
            options
            and answer_label in options
            and re.search(
                r"\b(?:not|never|without|does\s+not|doesn't|did\s+not|didn't|absent)\b",
                _semantic_query_text(query),
                re.IGNORECASE,
            )
        )
        native_negative_relation_supported = bool(
            negative_option_query
            and _has_native_negative_option_relation(
                answer_label=answer_label,
                options=options,
                steps=step_list,
                query=query,
            )
        )
        if native_negative_relation_supported:
            # A negative answer is supported by a complete native sequence or
            # an explicit selected-option absence.  Uncertain Observer claims
            # from earlier probes are not needed for the score and should not
            # dilute coverage once the sequence gate has passed.
            slots = [
                item for item in slots
                if not _slot_is_uncertain(item)
                and not _slot_is_synthetic_only(item)
                and _slot_has_executed_source(item, step_list)
            ]
            all_verified_slots = [
                item for item in slots
                if str(item.get("evidence_status") or "").lower() == "verified"
                and _slot_has_executed_source(item, step_list)
            ]
        answer_tokens = _content_tokens(answer)
        if options and answer_label in options:
            has_verified_answer_slot = any(
                str(item.get("evidence_status") or "").lower() == "verified"
                and not _slot_is_uncertain(item)
                and _slot_has_executed_source(item, step_list)
                and bool(answer_tokens.intersection(_content_tokens(f"{item.get('label', '')} {item.get('value', '')}")))
                for item in slots
            )
            if has_verified_answer_slot:
                relevant_slots = [
                    item for item in slots
                    if bool(item.get("logical conflict", item.get("logical_conflict", False)))
                    or (
                        str(item.get("evidence_status") or "").lower() == "verified"
                        and not _slot_is_uncertain(item)
                        and _slot_has_executed_source(item, step_list)
                        and bool(answer_tokens.intersection(_content_tokens(
                            f"{item.get('label', '')} {item.get('value', '')}"
                        )))
                    )
                ]
                if relevant_slots:
                    slots = relevant_slots
        verified = [
            item for item in slots
            if str(item.get("evidence_status", "")).lower() == "verified"
            and not _slot_is_uncertain(item)
            and _slot_has_executed_source(item, step_list)
        ]
        contradicted = [
            item for item in slots
            if not bool(item.get("context_only", False))
            and str(item.get("evidence_status") or "").strip().lower() == "verified"
            and _slot_has_executed_source(item, step_list)
            and not any(
                marker in str(item.get("uncertainty") or "").strip().lower()
                for marker in _UNCERTAINTY_MARKERS
            )
            and bool(item.get("logical conflict", item.get("logical_conflict", False)))
        ]
        if options and answer_label in options and contradicted:
            # A comparison task can legitimately collect a rejected
            # alternative (for example conflicting eye-color probes) while a
            # later direct probe settles the selected body part.  Let only
            # conflicts touching the selected option contribute to the global
            # contradiction score; premise mismatch and selected-option
            # comparison gates still inspect the complete evidence set.
            selected_tokens = _content_tokens(
                f"{answer_label} {options.get(answer_label, '')}"
            )
            contradicted = [
                item for item in contradicted
                if selected_tokens.intersection(_content_tokens(
                    " ".join(
                        str(item.get(key, "") or "")
                        for key in ("label", "target", "relation", "direction", "observed_value", "value")
                    )
                ))
            ]
        grounded = [item for item in verified if str(item.get("citation", "")).strip()]
        denominator = max(1, len(slots))
        coverage = len(verified) / denominator if slots else 0.0
        grounding = len(grounded) / denominator if slots else 0.0
        direct_frame_resolution = _has_direct_frame_resolution(
            answer=answer,
            query=query,
            verified_slots=verified,
            steps=step_list,
        )
        direct_frame_resolution = direct_frame_resolution or _has_native_conflict_resolution(
            answer=answer,
            query=query,
            verified_slots=verified,
            steps=step_list,
        )
        contradiction = len(contradicted) / denominator if slots else 0.0
        if direct_frame_resolution:
            slots = [item for item in slots if not _is_prior_textual_conflict(item)]
            verified = [
                item for item in slots
                if str(item.get("evidence_status", "")).lower() == "verified"
                and not _slot_is_uncertain(item)
                and _slot_has_executed_source(item, step_list)
            ]
            contradicted = [
                item
                for item in slots
                if not bool(item.get("context_only", False))
                and str(item.get("evidence_status") or "").strip().lower() == "verified"
                and _slot_has_executed_source(item, step_list)
                and not any(
                    marker in str(item.get("uncertainty") or "").strip().lower()
                    for marker in _UNCERTAINTY_MARKERS
                )
                and bool(item.get("logical conflict", item.get("logical_conflict", False)))
            ]
            denominator = max(1, len(slots))
            coverage = len(verified) / denominator if slots else 0.0
            grounding = (
                sum(1 for item in verified if str(item.get("citation", "")).strip())
                / denominator
                if slots
                else 0.0
            )
            contradiction = len(contradicted) / denominator if slots else 0.0
        families = _successful_families(step_list)
        normalized_basis = completion_basis.lower().replace(" ", "")
        basis_is_cross_modal = "text->visual" in normalized_basis or "visual->text" in normalized_basis
        cross_modal = 1.0 if (
            "cross_modal" in families or {"text", "visual"}.issubset(families) or basis_is_cross_modal
        ) else 0.0
        progress = min(1.0, 0.45 * coverage + 0.35 * grounding + 0.20 * cross_modal)

        missing: List[str] = []
        unresolved: List[str] = []
        option_mismatch = False
        premise_mismatch = _explicit_premise_mismatch(
            query=query,
            options=options,
            verified_slots=all_verified_slots,
            steps=step_list,
        )
        if premise_mismatch:
            unresolved.append(
                "direct evidence negates the question premise; no listed option is positively supported"
            )
        if _tool_returned_unlisted_action_value(
            query=query,
            options=options,
            steps=step_list,
        ):
            unresolved.append(
                "Video_Clip_QA returned an action-count value outside the listed options; do not map it to the nearest label"
            )
        if options and "order" in _claim_contract_requirements(query, self.thresholds):
            option_mismatch = _verified_order_has_no_listed_option(
                query=query,
                options=options,
                verified_slots=all_verified_slots,
            )
            if option_mismatch:
                unresolved.append(
                    "verified event order does not match any listed option; do not select the nearest sequence"
                )
        if not answer.strip():
            missing.append("non-empty answer candidate")
        if options and answer_label not in options:
            missing.append("final answer must select one listed option label")
        elif options and not _answer_matches_selected_option(
            answer=answer,
            answer_label=answer_label,
            options=options,
        ):
            unresolved.append("final answer text does not match the selected listed option")
        if options and _requires_option_comparison(query, self.thresholds):
            if not _has_option_comparison_evidence(
                answer_label=answer_label,
                options=options,
                verified_slots=verified,
                steps=step_list,
                query=query,
            ):
                unresolved.append("selected option lacks verified comparison against plausible alternatives")
            if _comparison_option_is_ambiguous(
                answer_label=answer_label,
                options=options,
                verified_slots=verified,
                query=query,
                steps=step_list,
            ):
                unresolved.append(
                    "comparison evidence supports multiple listed options without an explicit discriminator"
                )
        if options and _rationale_option_is_ambiguous(
            answer_label=answer_label,
            options=options,
            verified_slots=all_verified_slots,
            steps=step_list,
            query=query,
        ):
            unresolved.append(
                "rationale evidence supports multiple option outcomes; obtain an explicit discriminator before selecting a label"
            )
        claim_contract = _audit_claim_contract(
            query=query,
            verified_slots=all_verified_slots,
            steps=step_list,
            answer_label=answer_label,
            options=options,
            thresholds=self.thresholds,
        )
        for dimension in claim_contract.get("missing") or []:
            unresolved.append(f"claim contract missing: {dimension}")
        if _action_total_semantic_mismatch(query=query, verified_slots=verified):
            unresolved.append(
                "evidence reports a total or pre-existing count for an action-count question"
            )
        if options and not negative_option_query and _selected_option_is_negated(
            answer_label=answer_label, options=options, verified_slots=verified,
        ):
            unresolved.append("selected option is described as absent in verified evidence")
        if _count_evidence_conflict(
            query=query,
            verified_slots=verified,
            steps=step_list,
        ):
            unresolved.append(
                "direct count evidence conflicts across native answers or verified count slots; reconcile the windows before selecting a label"
            )
        if _incompatible_verified_claims(
            query=query,
            verified_slots=all_verified_slots,
            steps=step_list,
        ):
            unresolved.append(
                "incompatible verified direct claims for the same target disagree; resolve identity or time before selecting a label"
            )
        if not slots:
            missing.append("structured evidence claims")
        if coverage < float(self.thresholds.get("claim_coverage", 0.8)):
            unresolved.extend(str(item.get("label") or "unnamed claim") for item in slots if item not in verified)
            if not unresolved:
                unresolved.append("answer claim has no verified evidence")
        if grounding < float(self.thresholds.get("direct_grounding", 0.65)):
            missing.append("direct evidence citation")
        if bool(self.thresholds.get("require_answer_support", True)) and answer.strip():
            has_answer_support = _has_direct_answer_support(
                answer,
                verified,
                query=query,
                steps=step_list,
            )
            if options and "order" in claim_contract["required"]:
                has_answer_support = _has_order_answer_support(
                    answer_label=answer_label,
                    options=options,
                    query=query,
                    verified_slots=verified,
                    steps=step_list,
                )
            if native_negative_relation_supported:
                # The selected option is intentionally absent, so literal
                # token overlap with a verified slot is the wrong support
                # criterion.  The negative comparison helper has already
                # established the absence from a complete native sequence.
                has_answer_support = True
            if not has_answer_support:
                unresolved.append("answer candidate lacks directly matched verified evidence")
        if "dynamic_relation" in claim_contract["required"] or _query_asks_action_count(_semantic_query_text(query)):
            # A complete native negative sequence (for example a verified
            # Video_Summary listing every observed action) already proves that
            # the selected option is absent. Requiring a second
            # Video_Clip_QA call here would make the summary path impossible
            # to finish and would turn a tool preference into a hidden
            # benchmark contract.
            dynamic_support = native_negative_relation_supported or _has_dynamic_answer_support(
                answer, step_list, verified, query=query
            )
            if not dynamic_support:
                missing.append("direct temporal evidence supporting the selected action")
        if contradiction > float(self.thresholds.get("contradiction_score", 0.2)):
            missing.append("resolve contradictory evidence")
        require_cross_modal = bool(self.thresholds.get("require_cross_modal", True))
        if require_cross_modal and cross_modal == 0.0:
            missing.append("text and visual evidence chain")
        if bool(self.thresholds.get("require_completion_basis", False)) and not completion_basis.strip():
            missing.append("completion basis")

        can_finish = not missing and not unresolved
        feedback = (
            "Evidence verifier accepted termination."
            if can_finish
            else "Evidence verifier rejected termination; acquire or reconcile: " + ", ".join(missing + unresolved)
        )
        return VerificationResult(
            progress_score=round(progress, 6),
            claim_coverage=round(coverage, 6),
            direct_grounding=round(grounding, 6),
            contradiction_score=round(contradiction, 6),
            cross_modal_support=round(cross_modal, 6),
            can_finish=can_finish,
            premise_mismatch=premise_mismatch,
            option_mismatch=option_mismatch,
            missing_information=missing,
            unresolved_claims=unresolved,
            claim_contract=claim_contract,
            feedback=feedback,
        )
