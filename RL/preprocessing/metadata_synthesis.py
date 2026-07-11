"""
Metadata Synthesis Module
Generates titles, summaries, and tags for video segments using LLM.
"""
import json
import logging
import inspect
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class MetadataSynthesizer:
    """Synthesizes metadata (title, summary, tags) for video segments."""
    
    def __init__(self, llm_provider=None):
        """
        Initialize metadata synthesizer.
        
        Args:
            llm_provider: LLM provider for generating metadata
        """
        self.llm_provider = llm_provider
        self._is_stub = getattr(llm_provider, "__class__", type("X", (), {})).__name__ == "LocalProvider"
    
    async def synthesize_segment_metadata(
        self,
        caption: str,
        transcript: str,
        segment_id: str
    ) -> Dict[str, Any]:
        """
        Generate metadata for a single segment.
        
        Args:
            caption: Visual caption
            transcript: Audio transcript
            segment_id: Segment identifier
            
        Returns:
            Dictionary with title, summary, and tags
        """
        if not self.llm_provider:
            return self._fallback_metadata(caption, transcript)
        
        # Skip stub local provider to avoid noisy warnings and low-quality output
        if self._is_stub:
            return self._fallback_metadata(caption, transcript)
        
        prompt = self._build_synthesis_prompt(caption, transcript)
        
        try:
            response = await self._call_llm(prompt, max_tokens=200)
            metadata = self._parse_metadata_response(response, caption=caption, transcript=transcript)
        except Exception as e:
            logger.error(f"Error synthesizing metadata for {segment_id}: {e}")
            metadata = self._fallback_metadata(caption, transcript)
        
        return metadata
    
    def _build_synthesis_prompt(self, caption: str, transcript: str) -> str:
        """
        Build prompt for metadata synthesis.
        
        Args:
            caption: Visual caption
            transcript: Audio transcript
            
        Returns:
            Synthesis prompt
        """
        return f"""Based on the following video segment information, generate concise metadata in JSON format.

Requirements:
1. **Title** (max 10 words): Summarize the MAIN event or topic
2. **Summary** (max 50 words): Include key people, actions, objects, and context
3. **Tags** (3-5 keywords): Extract SPECIFIC nouns, verbs, or entities
   - Do NOT use generic words like "video", "presents", "shows", "begins", "the", "a", "an"
   - Focus on: people names, objects, actions, locations, concepts
   - Examples: "debt payment", "S&P 500", "investment strategy"

Video Caption:
{caption}

Audio Transcript:
{transcript}

Output format (JSON only, no explanation):
{{
  "title": "...",
  "summary": "...",
  "tags": ["tag1", "tag2", "tag3"]
}}

Generate metadata:"""
    
    async def _call_llm(self, prompt: str, **kwargs) -> str:
        """
        Call LLM provider safely whether it is sync or async.
        """
        generate = getattr(self.llm_provider, "generate", None)
        if not generate:
            raise RuntimeError("LLM provider has no generate method")

        if hasattr(self.llm_provider, "format_messages"):
            messages = self.llm_provider.format_messages(
                system_prompt="You generate concise metadata JSON with keys: title, summary, tags.",
                user_message=prompt,
                history=[]
            )
            result = generate(messages, **kwargs)
        else:
            result = generate(prompt, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
    
    def _parse_metadata_response(self, response: str, caption: str = "", transcript: str = "") -> Dict:
        """
        Parse LLM response into metadata structure.
        
        Args:
            response: LLM response
            
        Returns:
            Metadata dictionary
        """
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                metadata = json.loads(json_str)
                
                # Validate and clean
                return {
                    "title": metadata.get("title", "Unknown Scene")[:50],
                    "summary": metadata.get("summary", "")[:200],
                    "tags": metadata.get("tags", [])[:5]
                }
        except Exception as e:
            print(f"Error parsing metadata JSON: {e}")
        
        return self._fallback_metadata(caption, transcript)
    
    def _fallback_metadata(self, caption: str, transcript: str) -> Dict:
        """
        Generate simple fallback metadata.
        
        Args:
            caption: Visual caption
            transcript: Audio transcript
            
        Returns:
            Basic metadata dictionary
        """
        text = f"{caption} {transcript}".strip()
        words = text.lower().split()
        # Expanded stop words list to filter generic terms
        common_words = {
            # Articles & Pronouns
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'your', 'my', 'our',
            # Conjunctions & Prepositions  
            'and', 'or', 'but', 'if', 'then', 'because', 'as', 'since', 'while', 'when',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from', 'into', 'about', 'by',
            # Verbs (generic)
            'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
            'get', 'got', 'getting', 'make', 'makes', 'made', 'making',
            # Generic video-related words
            'video', 'videos', 'presents', 'shows', 'begins', 'starts', 'ends',
            'scene', 'segment', 'clip', 'footage', 'frame', 'image', 'visual',
            'depicts', 'displays', 'features', 'includes', 'contains',
            # Other common words
            'also', 'more', 'some', 'any', 'all', 'each', 'every', 'both', 'either', 'neither',
            'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down',
            'out', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'where',
            'why', 'how', 'what', 'which', 'who', 'whom', 'whose', 'when'
        }
        tags = []
        for token in words:
            token = token.strip(".,!?;:'\"()[]{}")
            if len(token) <= 3 or token in common_words:
                continue
            if token not in tags:
                tags.append(token)
            if len(tags) >= 5:
                break
        
        caption_clean = caption.strip()
        transcript_clean = transcript.strip()
        is_placeholder_caption = caption_clean.lower().startswith("visual caption unavailable")

        title_source = transcript_clean if is_placeholder_caption else (caption_clean or transcript_clean)
        if title_source:
            title = " ".join(title_source.split()[:10])[:80]
        else:
            title = "Video Segment"

        summary_source = transcript_clean if is_placeholder_caption and transcript_clean else text.strip()
        if summary_source:
            summary = summary_source[:220]
        else:
            summary = "No content available"

        return {
            "title": title,
            "summary": summary,
            "tags": tags
        }
    
    async def synthesize_all_segments(
        self,
        segments: Dict[str, Dict],
        captions: Dict[str, str],
        transcripts: Dict[str, str]
    ) -> Dict[str, Dict]:
        """
        Generate metadata for all segments.
        
        Args:
            segments: Segment information
            captions: Segment captions
            transcripts: Segment transcripts
            
        Returns:
            Enhanced segment data with metadata
        """
        enhanced_segments = {}
        
        for seg_id, seg_data in segments.items():
            caption = captions.get(seg_id, "")
            transcript = transcripts.get(seg_id, "")
            
            metadata = await self.synthesize_segment_metadata(caption, transcript, seg_id)
            
            enhanced_segments[seg_id] = {
                **seg_data,
                "caption": caption,
                "transcript": transcript,
                "title": metadata["title"],
                "summary": metadata["summary"],
                "tags": metadata["tags"]
            }
        
        return enhanced_segments
