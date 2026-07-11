"""
Knowledge Graph Construction Module
Builds scene-entity temporal knowledge graphs from video segments.
"""
import asyncio
import hashlib
import re
import json
import logging
import inspect
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


GRAPH_CACHE_VERSION = 2
GRAPH_TUPLE_DELIMITER = "<|>"
GRAPH_RECORD_DELIMITER = "##"
GRAPH_COMPLETION_DELIMITER = "<|COMPLETE|>"
GRAPH_CONTINUE_EXTRACTION_PROMPT = (
    "Some entities or relationships may still be missing. "
    "Add only the missing records using the same output format."
)
GRAPH_LOOP_CHECK_PROMPT = (
    "Are there still important missing entities or relationships grounded in the text? "
    "Answer only YES or NO."
)
DEFAULT_ENTITY_TYPES = (
    "person",
    "object",
    "location",
    "event",
    "organization",
    "concept",
)
_GENERIC_ENTITY_NAMES = {
    "someone", "person", "people", "man", "woman", "thing", "object", "objects",
    "scene", "video", "segment", "frame", "caption", "transcript", "audio",
}


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    name: str
    type: str
    description: str
    source_segments: List[str]
    time_segments: List[Tuple[str, str]] = field(default_factory=list)
    source_indices: List[str] = field(default_factory=list)
    modalities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.name,
            "label": self.name,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "source_segments": self.source_segments,
            "time_segments": self.time_segments,
            "source_indices": self.source_indices,
            "modalities": self.modalities
        }


@dataclass
class Relation:
    """Represents a relationship between entities."""
    source: str
    target: str
    relation_type: str
    description: str
    weight: float
    source_segments: List[str]
    time_range: Tuple[str, str] = ("0", "0")
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "relation": self.relation_type or self.description,
            "description": self.description,
            "weight": self.weight,
            "source_segments": self.source_segments,
            "time_range": self.time_range
        }


class KnowledgeGraphBuilder:
    """Builds temporal knowledge graphs from video content."""
    
    def __init__(self, llm_provider=None):
        """
        Initialize knowledge graph builder.
        
        Args:
            llm_provider: LLM provider for entity extraction
        """
        self.llm_provider = llm_provider
        self._is_stub = getattr(llm_provider, "__class__", type("X", (), {})).__name__ == "LocalProvider"
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.segments_data: Dict[str, Dict] = {}
        self.segment_event_nodes: Dict[str, str] = {}
        self._relation_index: set = set()
        self._relation_lookup: Dict[Tuple[str, str, str], Relation] = {}
    
    def register_segment(self, segment_id: str, segment_data: Dict[str, Any]) -> None:
        """Store segment metadata for fallback graph construction."""
        self.segments_data[segment_id] = segment_data or {}
    
    async def extract_entities_from_segment(
        self,
        segment_content: str,
        segment_id: str,
        max_gleaning: int = 1
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relationships from a video segment.
        
        Args:
            segment_content: Segment caption and transcript
            segment_id: Unique segment identifier
            max_gleaning: Number of extraction iterations
            
        Returns:
            Tuple of (entities, relations)
        """
        entities: List[Entity] = []
        relations: List[Relation] = []
        
        segment_meta = self.segments_data.get(segment_id, {})
        time_tuple = self._parse_time_range(segment_meta.get("time", "0-0"))

        if self._is_stub:
            # Skip stub local provider; go straight to fallback graph
            fallback_entities, fallback_relations = self._generate_fallback_segment_graph(segment_id)
            return fallback_entities, fallback_relations

        if self.llm_provider:
            extraction_prompt = self._build_extraction_prompt(segment_content)
            
            for _ in range(max_gleaning):
                try:
                    response = await self._call_llm(extraction_prompt)
                    response = await self._maybe_glean_missing_records(
                        extraction_prompt,
                        response,
                        max_gleaning=max(1, int(max_gleaning or 1)),
                    )
                    parsed_entities, parsed_relations = self._parse_extraction_response(
                        response, segment_id
                    )
                    entities.extend(parsed_entities)
                    relations.extend(parsed_relations)
                except Exception as e:
                    logger.warning(f"Error extracting entities from segment {segment_id}: {e}")
        
        # Enrich extracted items with temporal/source info
        for ent in entities:
            if not ent.time_segments:
                ent.time_segments.append(time_tuple)
            if not ent.source_indices:
                ent.source_indices.append(segment_id)
            if not ent.modalities:
                ent.modalities.append("llm")
        for rel in relations:
            if rel.time_range == ("0", "0"):
                rel.time_range = time_tuple

        if not entities and not relations:
            fallback_entities, fallback_relations = self._generate_fallback_segment_graph(segment_id)
            entities.extend(fallback_entities)
            relations.extend(fallback_relations)
        
        return entities, relations

    async def _call_llm(self, prompt: str, history: Optional[List[Dict[str, str]]] = None, **kwargs) -> str:
        """Call LLM provider safely whether it is sync or async."""
        generate = getattr(self.llm_provider, "generate", None)
        if not generate:
            raise RuntimeError("LLM provider has no generate method")

        format_messages = getattr(self.llm_provider, "format_messages", None)
        if callable(format_messages):
            payload = format_messages(
                system_prompt=(
                    "Extract entities and relationships from video segment content. "
                    "Return only the requested pipe-delimited lines."
                ),
                user_message=prompt,
                history=history,
            )
        else:
            payload = prompt

        result = generate(payload, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _maybe_glean_missing_records(
        self,
        extraction_prompt: str,
        first_response: str,
        *,
        max_gleaning: int = 1,
    ) -> str:
        if max_gleaning <= 1 or not str(first_response or "").strip():
            return str(first_response or "")

        final_response = str(first_response or "")
        history = [
            {"role": "user", "content": extraction_prompt},
            {"role": "assistant", "content": final_response},
        ]
        for glean_index in range(max(0, max_gleaning - 1)):
            gleaned = await self._call_llm(
                GRAPH_CONTINUE_EXTRACTION_PROMPT,
                history=history,
            )
            gleaned_text = str(gleaned or "").strip()
            if gleaned_text:
                final_response += "\n" + gleaned_text
                history.extend(
                    [
                        {"role": "user", "content": GRAPH_CONTINUE_EXTRACTION_PROMPT},
                        {"role": "assistant", "content": gleaned_text},
                    ]
                )
            if glean_index == max_gleaning - 2:
                break
            loop_check = await self._call_llm(
                GRAPH_LOOP_CHECK_PROMPT,
                history=history,
            )
            normalized_check = str(loop_check or "").strip().strip('"').strip("'").lower()
            if normalized_check != "yes":
                break
        return final_response
    
    def _build_extraction_prompt(self, content: str) -> str:
        """
        Build prompt for entity extraction.
        
        Args:
            content: Segment content
            
        Returns:
            Extraction prompt
        """
        entity_types = ", ".join(DEFAULT_ENTITY_TYPES)
        return f"""-Goal-
Given a text document describing a video segment and a list of entity types, identify the entities of those types and the relationships among them.

-Steps-
1. Identify useful entities from the text.
- entity_name: preserve the original surface form when possible
- entity_type: one of [{entity_types}]
- entity_description: a concise description grounded in the segment text
Format each entity as:
("entity"{GRAPH_TUPLE_DELIMITER}<entity_name>{GRAPH_TUPLE_DELIMITER}<entity_type>{GRAPH_TUPLE_DELIMITER}<entity_description>)

2. Identify clearly supported relationships between the entities.
- source_entity
- target_entity
- relation_type: a short canonical label such as TEMPORAL_NEXT / INTERACTION / HOLDING / PART_OF / MENTIONS / RELATED_TO
- relationship_description
- relationship_strength: integer or float strength score
Format each relationship as:
("relationship"{GRAPH_TUPLE_DELIMITER}<source_entity>{GRAPH_TUPLE_DELIMITER}<target_entity>{GRAPH_TUPLE_DELIMITER}<relation_type>{GRAPH_TUPLE_DELIMITER}<relationship_description>{GRAPH_TUPLE_DELIMITER}<relationship_strength>)

3. Return a single list joined by {GRAPH_RECORD_DELIMITER}.
4. End with {GRAPH_COMPLETION_DELIMITER}.

Entity_types: [{entity_types}]
Text: {content}
Output:"""
    
    def _parse_extraction_response(
        self,
        response: str,
        segment_id: str
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Parse LLM response into entities and relations.
        
        Args:
            response: LLM response text
            segment_id: Source segment ID
            
        Returns:
            Tuple of (entities, relations)
        """
        entities = []
        relations = []
        
        records = self._split_extraction_records(response)
        for record in records:
            parts = self._split_record_attributes(record)
            if not parts:
                continue

            record_type = str(parts[0]).strip().strip('"').strip("'").lower()
            if record_type == "entity" and len(parts) >= 4:
                raw_name = str(parts[1]).strip()
                entity = Entity(
                    name=self._normalize_entity_name(raw_name),
                    type=self._normalize_entity_type(str(parts[2]).strip()),
                    description=self._normalize_description(str(parts[3]).strip()),
                    source_segments=[segment_id],
                )
                if self._is_valid_entity_name(entity.name):
                    entities.append(entity)

            elif record_type == "relationship" and len(parts) >= 4:
                raw_relation_type = ""
                raw_relation_description = ""
                raw_weight = ""
                if len(parts) >= 6:
                    raw_relation_type = str(parts[3]).strip()
                    raw_relation_description = str(parts[4]).strip()
                    raw_weight = str(parts[5]).strip()
                else:
                    raw_relation_description = str(parts[3]).strip()
                    raw_weight = str(parts[4]).strip() if len(parts) > 4 else ""
                relation_description = self._normalize_description(raw_relation_description)
                weight = float(raw_weight) if raw_weight and self._is_float(raw_weight) else 1.0
                relation = Relation(
                    source=self._normalize_entity_name(str(parts[1]).strip()),
                    target=self._normalize_entity_name(str(parts[2]).strip()),
                    relation_type=self._normalize_entity_type(raw_relation_type) if raw_relation_type else self._infer_relation_type(relation_description),
                    description=relation_description,
                    weight=weight,
                    source_segments=[segment_id]
                )
                if self._is_valid_entity_name(relation.source) and self._is_valid_entity_name(relation.target):
                    relations.append(relation)
        
        return entities, relations

    def _split_extraction_records(self, response: str) -> List[str]:
        text = str(response or "").strip()
        if not text:
            return []

        tuple_records = re.findall(r"\((.*?)\)", text, flags=re.DOTALL)
        if tuple_records:
            return [record.strip() for record in tuple_records if record.strip()]

        raw_records = re.split(
            rf"{re.escape(GRAPH_RECORD_DELIMITER)}|{re.escape(GRAPH_COMPLETION_DELIMITER)}|\n+",
            text,
        )
        return [record.strip() for record in raw_records if record.strip()]

    def _split_record_attributes(self, record: str) -> List[str]:
        text = str(record or "").strip()
        if not text:
            return []
        if GRAPH_TUPLE_DELIMITER in text:
            parts = text.split(GRAPH_TUPLE_DELIMITER)
        else:
            parts = text.split("|")
        return [part.strip().strip('"').strip("'") for part in parts if str(part).strip()]
    
    def _is_float(self, value: str) -> bool:
        """Check if string is a valid float."""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _normalize_entity_name(self, value: str) -> str:
        normalized = re.sub(r"\s+", " ", str(value or "").strip())
        normalized = re.sub(r"^[\"'`]+|[\"'`]+$", "", normalized)
        normalized = re.sub(r"[^A-Za-z0-9 _-]+", "", normalized)
        return normalized.upper().strip(" _-")

    def _normalize_entity_type(self, value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
        if not normalized:
            return "CONCEPT"
        return normalized.upper()

    def _normalize_description(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    def _is_valid_entity_name(self, value: str) -> bool:
        normalized = str(value or "").strip()
        lowered = normalized.lower()
        if not normalized or len(normalized) < 2:
            return False
        if lowered in _GENERIC_ENTITY_NAMES:
            return False
        if normalized.startswith("EVENT_"):
            return True
        if lowered.isdigit():
            return False
        return True

    def _infer_relation_type(self, description: str) -> str:
        text = str(description or "").lower()
        relation_rules = [
            ("TEMPORAL_NEXT", ("before", "after", "next", "then", "follows", "followed by")),
            ("CROSS_MODAL_ALIGN", ("same entity", "same object", "aligned", "corresponds to")),
            ("CO_OCCUR", ("co-occur", "appear together", "same scene", "same segment")),
            ("INTERACTION", ("interact", "talk", "speak", "argue", "conversation")),
            ("HOLDING", ("hold", "carry", "grab", "pick up")),
            ("MOVING_TO", ("move to", "walk to", "run to", "approach", "go toward")),
            ("PART_OF", ("part of", "belongs to", "inside", "within")),
            ("MENTIONS", ("mentions", "refers to", "shows", "depicts")),
        ]
        for relation_type, keywords in relation_rules:
            if any(keyword in text for keyword in keywords):
                return relation_type
        return "RELATED_TO"
    
    def add_entity(self, entity: Entity) -> None:
        """
        Add or merge entity into graph.
        
        Args:
            entity: Entity to add
        """
        entity.name = self._normalize_entity_name(entity.name)
        entity.type = self._normalize_entity_type(entity.type)
        entity.description = self._normalize_description(entity.description)
        if not self._is_valid_entity_name(entity.name):
            return
        self._ensure_entity_temporal_fields(entity)

        if entity.name in self.entities:
            # Merge with existing entity
            existing = self.entities[entity.name]
            if entity.description and entity.description not in existing.description.split(" | "):
                existing.description = (
                    f"{existing.description} | {entity.description}"
                    if existing.description
                    else entity.description
                )
            self._extend_unique(existing.source_segments, entity.source_segments)
            self._extend_unique(existing.time_segments, entity.time_segments)
            self._extend_unique(existing.source_indices, entity.source_indices)
            # Merge modalities without duplicates
            for mod in entity.modalities:
                if mod not in existing.modalities:
                    existing.modalities.append(mod)
            self._ensure_entity_temporal_fields(existing)
        else:
            self.entities[entity.name] = entity
    
    def add_relation(self, relation: Relation) -> None:
        """
        Add relationship to graph.
        
        Args:
            relation: Relation to add
        """
        relation.source = self._normalize_entity_name(relation.source)
        relation.target = self._normalize_entity_name(relation.target)
        relation.relation_type = self._normalize_entity_type(relation.relation_type)
        relation.description = self._normalize_description(relation.description)
        if not self._is_valid_entity_name(relation.source) or not self._is_valid_entity_name(relation.target):
            return
        if relation.source == relation.target and relation.relation_type != "TEMPORAL_NEXT":
            return
        if relation.time_range == ("0", "0") and relation.source_segments:
            inferred = self._infer_time_range_from_segment(relation.source_segments[0])
            relation.time_range = inferred
        relation_key = (
            relation.source,
            relation.target,
            relation.relation_type,
        )
        existing = self._relation_lookup.get(relation_key)
        if existing is not None:
            existing.weight = max(float(existing.weight or 0.0), float(relation.weight or 0.0))
            self._extend_unique(existing.source_segments, relation.source_segments)
            if relation.time_range not in (("0", "0"), None) and relation.time_range != existing.time_range:
                current_start, current_end = existing.time_range
                next_start, next_end = relation.time_range
                merged_start = current_start
                merged_end = current_end
                if self._to_seconds(str(next_start)) < self._to_seconds(str(current_start)):
                    merged_start = next_start
                if self._to_seconds(str(next_end)) > self._to_seconds(str(current_end)):
                    merged_end = next_end
                existing.time_range = (
                    str(merged_start),
                    str(merged_end),
                )
            if relation.description and relation.description not in existing.description.split(" | "):
                existing.description = (
                    f"{existing.description} | {relation.description}"
                    if existing.description
                    else relation.description
                )
            return
        self._relation_index.add(relation_key)
        self._relation_lookup[relation_key] = relation
        self.relations.append(relation)

    def _infer_time_range_from_segment(self, segment_id: str) -> Tuple[str, str]:
        seg = self.segments_data.get(str(segment_id), {})
        return self._parse_time_range(seg.get("time", "0-0"))

    def _ensure_entity_temporal_fields(self, entity: Entity) -> None:
        if not entity.source_segments:
            return
        if not entity.source_indices:
            entity.source_indices.extend(entity.source_segments)
        if not entity.time_segments:
            first_seg = str(entity.source_segments[0])
            entity.time_segments.append(self._infer_time_range_from_segment(first_seg))

    def _extend_unique(self, target: List[Any], values: List[Any]) -> None:
        for value in values or []:
            if value not in target:
                target.append(value)
    
    def build_temporal_connections(self, segment_order: List[str]) -> None:
        """
        Build temporal connections between entities in adjacent segments.
        
        Args:
            segment_order: Ordered list of segment IDs
        """
        # Group entities by segment
        segment_entities: Dict[str, List[str]] = defaultdict(list)
        for entity_name, entity in self.entities.items():
            for seg_id in entity.source_segments:
                segment_entities[seg_id].append(entity_name)
        
        # Create temporal relations for entities appearing in adjacent segments
        for i in range(len(segment_order) - 1):
            curr_seg = segment_order[i]
            next_seg = segment_order[i + 1]
            
            curr_entities = set(segment_entities.get(curr_seg, []))
            next_entities = set(segment_entities.get(next_seg, []))
            
            # Find entities that appear in both segments
            common_entities = curr_entities & next_entities
            
            for entity in common_entities:
                # Add temporal continuation relation
                temporal_relation = Relation(
                    source=entity,
                    target=entity,
                    relation_type="TEMPORAL_NEXT",
                    description=f"Continues from {curr_seg} to {next_seg}",
                    weight=1.0,
                    source_segments=[curr_seg, next_seg]
                )
                self.add_relation(temporal_relation)
    
    def export_graph(self) -> Dict[str, Any]:
        """
        Export knowledge graph to dictionary format.
        
        Returns:
            Graph data structure
        """
        clean_entities: Dict[str, Dict[str, Any]] = {}
        for name, entity in self.entities.items():
            self._ensure_entity_temporal_fields(entity)
            payload = entity.to_dict()
            if not payload.get("source_segments"):
                continue
            clean_entities[name] = payload

        clean_relations: List[Dict[str, Any]] = []
        for rel in self.relations:
            if rel.source == rel.target and rel.relation_type != "TEMPORAL_NEXT":
                continue
            if rel.source not in clean_entities or rel.target not in clean_entities:
                continue
            rel_dict = rel.to_dict()
            clean_relations.append(rel_dict)
        clean_relations.sort(
            key=lambda item: (
                str(item.get("source", "")),
                str(item.get("relation_type", "")),
                str(item.get("target", "")),
                str(item.get("time_range", "")),
            )
        )

        return {
            "entities": clean_entities,
            "relations": clean_relations,
        }
    
    def save_graph(self, output_path: str) -> None:
        """
        Save knowledge graph to JSON file.
        
        Args:
            output_path: Output file path
        """
        graph_data = self.export_graph()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

    def _generate_fallback_segment_graph(self, segment_id: str) -> Tuple[List[Entity], List[Relation]]:
        """Generate entities and relations directly from segment metadata when LLM is unavailable."""
        segment = self.segments_data.get(segment_id, {})
        if not segment:
            return [], []

        time_range = segment.get("time", "0-0")
        time_tuple = self._parse_time_range(time_range)
        caption = segment.get("caption", "")
        transcript = segment.get("transcript", "")
        tags = segment.get("tags", [])
        summary = segment.get("summary") or caption or transcript
        summary = summary.strip() or "Scene description unavailable"

        event_name = f"EVENT_{segment_id}"
        event_entity = Entity(
            name=event_name,
            type="EVENT",
            description=summary[:300],
            source_segments=[segment_id],
            time_segments=[time_tuple],
            source_indices=[segment_id]
        )

        supporting_entities: List[Entity] = []
        relations: List[Relation] = []

        keywords = tags or self._extract_keywords(f"{caption} {transcript}")
        for keyword in keywords:
            entity_name = keyword.upper()
            supporting_entity = Entity(
                name=entity_name,
                type="CONCEPT",
                description=f"Concept extracted from segment {segment_id}: {keyword}",
                source_segments=[segment_id],
                time_segments=[time_tuple],
                source_indices=[segment_id]
            )
            supporting_entities.append(supporting_entity)

            relations.append(
                Relation(
                    source=event_name,
                    target=entity_name,
                    relation_type="MENTIONS",
                    description=f"Event {event_name} mentions {entity_name}",
                    weight=0.5,
                    source_segments=[segment_id],
                    time_range=time_tuple
                )
            )

        # Link sequential events for temporal continuity
        prev_event = self.segment_event_nodes.get("prev")
        if prev_event:
            relations.append(
                Relation(
                    source=prev_event,
                    target=event_name,
                    relation_type="TEMPORAL_NEXT",
                    description="Automatically generated temporal adjacency",
                    weight=0.4,
                    source_segments=[segment_id],
                    time_range=time_tuple
                )
            )
        self.segment_event_nodes["prev"] = event_name

        entities = [event_entity] + supporting_entities
        return entities, relations

    def _harvest_text_entities(
        self,
        text: str,
        segment_id: str,
        time_tuple: Tuple[str, str],
        modality: str,
    ) -> List[Entity]:
        """Simple keyword-based entity harvesting from text, tagged by modality."""
        keywords = self._extract_keywords(text)
        entities: List[Entity] = []
        for kw in keywords:
            entities.append(
                Entity(
                    name=kw.upper(),
                    type="CONCEPT",
                    description=f"Keyword from segment {segment_id}: {kw}",
                    source_segments=[segment_id],
                    time_segments=[time_tuple],
                    source_indices=[segment_id],
                    modalities=[modality],
                )
            )
        return entities

    def add_cooccurrence_relations(self, entities: List[Entity], segment_id: str, time_tuple: Tuple[str, str]) -> None:
        """Create co-occurrence relations within a single segment."""
        event_nodes = [e for e in entities if e.type == "EVENT" or e.name.startswith("EVENT_")]
        concept_nodes = [e for e in entities if e not in event_nodes]
        if not event_nodes:
            event_nodes = entities[:1]

        edge_cap = 20
        edge_count = 0
        for event in event_nodes:
            for concept in concept_nodes:
                if edge_count >= edge_cap:
                    break
                relation = Relation(
                    source=event.name,
                    target=concept.name,
                    relation_type="CO_OCCUR",
                    description=f"Co-occur in segment {segment_id}",
                    weight=0.6,
                    source_segments=[segment_id],
                    time_range=time_tuple,
                )
                self.add_relation(relation)
                edge_count += 1
            if edge_count >= edge_cap:
                break

    def add_interaction_relations(
        self,
        entities: List[Entity],
        segment_id: str,
        time_tuple: Tuple[str, str],
        text: str,
    ) -> None:
        """Create lightweight interaction/holding relations based on text cues."""
        cues = {
            "INTERACTION": ["argu", "fight", "conflict", "push", "pull", "talk", "speak"],
            "HOLDING": ["hold", "grab", "carry", "lift"],
            "VIOLENCE": ["shoot", "gun", "fire", "hit", "punch", "kick"],
        }
        text_lower = text.lower()
        matched = []
        for rel_type, keywords in cues.items():
            if any(k in text_lower for k in keywords):
                matched.append(rel_type)
        if not matched:
            return

        event_nodes = [e for e in entities if e.type == "EVENT" or e.name.startswith("EVENT_")]
        concept_nodes = [e for e in entities if e not in event_nodes]
        if not event_nodes:
            event_nodes = entities[:1]

        edge_cap = 20
        edge_count = 0
        for event in event_nodes:
            for concept in concept_nodes:
                if edge_count >= edge_cap:
                    break
                for rel_type in matched:
                    relation = Relation(
                        source=event.name,
                        target=concept.name,
                        relation_type=rel_type,
                        description=f"{rel_type} inferred in segment {segment_id}",
                        weight=0.7,
                        source_segments=[segment_id],
                        time_range=time_tuple,
                    )
                    self.add_relation(relation)
                edge_count += 1
            if edge_count >= edge_cap:
                break

    def align_modal_entities(
        self,
        entities: List[Entity],
        time_tuple: Tuple[str, str],
        segment_id: str,
        iou_threshold: float = 0.2,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Merge entities with the same name across modalities when their time ranges overlap.
        Returns merged entities and alignment relations.
        """
        merged: Dict[str, Entity] = {}
        align_relations: List[Relation] = []

        for ent in entities:
            key = ent.name
            if key not in merged:
                merged[key] = ent
                continue

            existing = merged[key]
            # Check temporal IOU
            iou = self._time_iou(existing.time_segments, ent.time_segments)
            if iou < iou_threshold:
                # keep both if they don't overlap sufficiently
                new_key = f"{ent.name}_{len(merged)}"
                merged[new_key] = ent
                continue

            # Merge modalities/time/indices
            for mod in ent.modalities:
                if mod not in existing.modalities:
                    existing.modalities.append(mod)
            existing.source_segments.extend(ent.source_segments)
            existing.source_indices.extend(ent.source_indices)
            existing.time_segments.extend(ent.time_segments)
            existing.description += f" | {ent.description}"

            if existing.name != ent.name:
                align_relations.append(
                    Relation(
                        source=existing.name,
                        target=ent.name,
                        relation_type="CROSS_MODAL_ALIGN",
                        description=f"Aligned modalities for {ent.name} in segment {segment_id}",
                        weight=0.7,
                        source_segments=[segment_id],
                        time_range=time_tuple,
                    )
                )

        return list(merged.values()), align_relations

    def _time_iou(self, segs_a: List[Tuple[str, str]], segs_b: List[Tuple[str, str]]) -> float:
        """Compute max IOU between two sets of time ranges."""
        max_iou = 0.0
        for a in segs_a:
            for b in segs_b:
                iou = self._range_iou(a, b)
                if iou > max_iou:
                    max_iou = iou
        return max_iou

    def _range_iou(self, a: Tuple[str, str], b: Tuple[str, str]) -> float:
        try:
            a_start, a_end = self._to_seconds(a[0]), self._to_seconds(a[1])
            b_start, b_end = self._to_seconds(b[0]), self._to_seconds(b[1])
        except Exception:
            return 0.0
        inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
        union = max(a_end, b_end) - min(a_start, b_start)
        if union <= 0:
            return 0.0
        return inter / union

    def _to_seconds(self, t: str) -> float:
        """Parse time string (s) or mm:ss to seconds."""
        if ":" in t:
            parts = t.split(":")
            seconds = 0.0
            for p in parts:
                seconds = seconds * 60 + float(p)
            return seconds
        return float(t)

    def _parse_time_range(self, time_range: str) -> Tuple[str, str]:
        """Parse time range string into tuple."""
        if '-' in time_range:
            start, end = time_range.split('-', 1)
            return start.strip(), end.strip()
        return ("0", "0")

    def _extract_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction for fallback nodes."""
        cleaned_text = re.sub(r"\[[0-9]+\.[0-9]+s\s*->\s*[0-9]+\.[0-9]+s\]", " ", text)
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", cleaned_text)
        common_words = {
            "the", "and", "with", "from", "that", "this", "have", "said", "there", "here", "they",
            "where", "when", "what", "which", "who", "whom", "whose", "able", "start", "would", "could",
            "should", "about", "into", "than", "then", "also", "very", "much", "more", "most", "some",
            "many", "such", "your", "their", "them", "were", "been", "being", "will", "shall", "might",
            "must", "does", "did", "done", "over", "under", "across", "after", "before", "during",
            "visual", "caption", "unavailable", "sampled", "frames", "frame", "transcript", "segment",
            "rather", "living", "long", "simple", "better", "since", "going", "back", "just", "next",
            "while", "almost", "possible", "show", "take", "down", "means", "extra", "instead", "thank"
        }
        token_freq: Dict[str, int] = {}
        for token in tokens:
            lower = token.lower()
            if len(lower) < 4 or lower in common_words:
                continue
            if lower.isdigit():
                continue
            token_freq[lower] = token_freq.get(lower, 0) + 1

        sorted_tokens = sorted(token_freq.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
        return [token for token, _ in sorted_tokens[:5]]


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _segment_source_signature(segments: Dict[str, Dict[str, Any]]) -> str:
    normalized: List[Dict[str, Any]] = []
    for seg_id, payload in sorted((segments or {}).items(), key=lambda item: str(item[0])):
        if not isinstance(payload, dict):
            continue
        normalized.append(
            {
                "segment_id": str(seg_id),
                "time": str(payload.get("time", "") or ""),
                "caption": str(payload.get("caption", "") or ""),
                "transcript": str(payload.get("transcript", "") or ""),
            }
        )
    raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _segment_sort_key(segment_id: Any) -> Tuple[int, Any]:
    text = str(segment_id)
    return (0, int(text)) if text.isdigit() else (1, text)


async def build_knowledge_graph_payload(
    segments: Dict[str, Dict[str, Any]],
    *,
    llm_provider: Any = None,
    max_gleaning: int = 1,
) -> Dict[str, Any]:
    builder = KnowledgeGraphBuilder(llm_provider=llm_provider)
    for seg_id, seg_data in (segments or {}).items():
        if isinstance(seg_data, dict):
            builder.register_segment(str(seg_id), seg_data)

    segment_order = sorted((segments or {}).keys(), key=_segment_sort_key)
    for seg_id in segment_order:
        segment_payload = dict((segments or {}).get(seg_id) or {})
        caption = str(segment_payload.get("caption", "") or "")
        transcript = str(segment_payload.get("transcript", "") or "")
        content = f"Caption: {caption}\nTranscript: {transcript}"
        entities, relations = await builder.extract_entities_from_segment(
            content,
            str(seg_id),
            max_gleaning=max(1, int(max_gleaning or 1)),
        )
        time_tuple = builder._parse_time_range(segment_payload.get("time", "0-0"))
        entities.extend(
            builder._harvest_text_entities(caption, str(seg_id), time_tuple, modality="visual")
        )
        entities.extend(
            builder._harvest_text_entities(transcript, str(seg_id), time_tuple, modality="audio")
        )
        entities, align_relations = builder.align_modal_entities(
            entities,
            time_tuple,
            str(seg_id),
            iou_threshold=0.2,
        )
        builder.add_interaction_relations(
            entities,
            str(seg_id),
            time_tuple,
            text=f"{caption} {transcript}",
        )
        builder.add_cooccurrence_relations(entities, str(seg_id), time_tuple)
        relations.extend(align_relations)

        for entity in entities:
            builder.add_entity(entity)
        for relation in relations:
            builder.add_relation(relation)

    builder.build_temporal_connections([str(seg_id) for seg_id in segment_order])
    return builder.export_graph()


def _run_async_in_new_thread(coro: Any) -> Any:
    container: Dict[str, Any] = {}
    error_box: Dict[str, BaseException] = {}

    def _target() -> None:
        try:
            container["result"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - sync bridge safety
            error_box["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()
    if "error" in error_box:
        raise error_box["error"]
    return container.get("result")


def ensure_knowledge_graph_for_processed_video(
    video_dir: Path | str,
    *,
    llm_provider: Any = None,
    max_gleaning: int = 1,
) -> Dict[str, Any]:
    processed_dir = Path(video_dir)
    segments_path = processed_dir / "segments.json"
    kg_path = processed_dir / "knowledge_graph.json"
    if not segments_path.exists():
        raise FileNotFoundError(f"Missing segments.json for knowledge-graph build: {segments_path}")

    segments = _load_json(segments_path) or {}
    if not isinstance(segments, dict):
        raise ValueError(f"Invalid segments payload in {segments_path}")

    source_signature = _segment_source_signature(segments)
    if kg_path.exists():
        try:
            payload = _load_json(kg_path) or {}
            meta = payload.get("_meta") if isinstance(payload, dict) else {}
            if (
                isinstance(payload, dict)
                and isinstance(meta, dict)
                and int(meta.get("graph_cache_version", 0) or 0) == GRAPH_CACHE_VERSION
                and str(meta.get("source_signature", "") or "") == source_signature
            ):
                return payload
        except Exception as exc:
            logger.warning("Failed to load cached knowledge graph from %s: %s", kg_path, exc)

    created_provider = False
    if llm_provider is None:
        try:
            from config.settings import settings
            from core.llm import get_llm_provider

            llm_provider = get_llm_provider(
                model_name=settings.preprocess_model_path,
                model_role="preprocess",
            )
            created_provider = True
        except Exception as exc:
            logger.warning("Falling back to heuristic graph build because preprocess LLM could not be loaded: %s", exc)
            llm_provider = None

    try:
        graph_payload = _run_async_in_new_thread(
            build_knowledge_graph_payload(
                segments,
                llm_provider=llm_provider,
                max_gleaning=max_gleaning,
            )
        )
    finally:
        if created_provider:
            try:
                from runtime.session import runtime_session

                runtime_session.release_providers_for_roles("preprocess")
            except Exception:
                pass

    result = dict(graph_payload or {})
    result["_meta"] = {
        "graph_cache_version": GRAPH_CACHE_VERSION,
        "source_signature": source_signature,
        "segment_count": len(segments),
    }
    _dump_json(kg_path, result)
    return result
