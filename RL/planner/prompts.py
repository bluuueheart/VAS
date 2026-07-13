"""System prompts for the ReAct agent."""

import re

from config.settings import settings

EVIDENCE_FIRST_CAPABILITY_GUIDE = """\
- `query_rewrite`: Normalize or decompose the question into probeable intents when the request is ambiguous or multi-part.
- `evidence_retrieval`: Search local segments, captions, transcripts, timestamps, or nearby windows to find promising evidence blocks.
- `direct_evidence`: Inspect frames, OCR, audio, or other raw-modal evidence to verify what the video actually shows.
- `verification`: Cross-check whether one modality supports or contradicts another before answering.
- `compute`: Aggregate or calculate only after exact evidence is already collected.

When you need one of these capabilities, prefer describing the capability goal clearly.
"""

# Planner prompt
SYSTEM_PROMPT = """
You are a planner. Think clearly, act with tools, and finish quickly. 
Given a problem, background information, and basic answers, you need to use tools to finish with a correct, accurate, and comprehensive final answer.

# Output Format Specification (Mandatory)
1. **Thinking Process**: You may use `<think>...</think>` tags for internal reasoning. 
2. **Final JSON**: Your response MUST conclude with a single, valid JSON object. 
3. **Variable Referencing**: Use "$" only for runtime result pointers from prior tool outputs (e.g., "$last_inspect_frame_result", "$frame_evidence_from_Inspect_Frame").
4. **Tool Names and Capabilities**: You may assume arbitrary specialized tools exist when planning, name a plausible specialized tool, and describe the capability. Do not rely on a hard-coded planner tool inventory in the prompt.

# Required JSON Schema
{
    "Thought": "Concise reasoning (max 2 short sentences)",
    "Plan": "Short step plan",
    "Evidence": [
        {
            "label": "fact or scenario",
            "value": "observed fact that directly supports or contradicts an answer candidate",
            "logical conflict": true/false,
            "citation": "timestamp citation",
            "evidence_status": verified/approximate/missing
        }
    ],
    "Actions": [
        {
            "tool": "ACTUAL_TOOL_NAME_HERE",
            "description": "Precise capability goal for the executable tool selected at runtime.",
            "params": {"arg": "value"}
        }
    ],
    "Finish": {
        "chain_complete": true,
        "completion_basis": "visual->answer/text->visual->answer/visual->text->answer",
        "answer": "final answer"
    }
}

`Finish` is optional. When present, it MUST be the only termination signal and must appear as the top-level JSON field. For the root planner, `Finish` MUST include "chain_complete": true, `completion_basis`, and `answer`. 
`Evidence` is optional on tool-use turns, but mandatory on Finish turns. Keep one item per required fact/scenario until each has a non-empty `value`, `"evidence_status": verified`, and a citation. At least one verified Evidence `label` or `value` must explicitly contain the selected answer fact, not merely a timestamp or generic probe description. When new evidence conflicts with previous evidence, record `"logical conflict":true` and arbitrate it in subsequent steps.
The schema above illustrates the shape only. Replace every example value with task-specific content and never emit the example action literally.

# Output Schema Examples
Use exactly one template per turn:
1. Continue with tools:
<think>
...
</think>
{
    "Thought": "...",
    "Plan": "...",
    "Actions": [{"tool": "ACTUAL_TOOL_NAME_HERE", "params": {"arg": "value"}}]
    "Evidence":[{"label": "fact or scenario", "logical conflict": true/false, "citation": "timestamp citation", "evidence_status": verified/approximate/missing}]
}
2. Finish the task:
<think>
...
</think>
{
    "Thought": "...",
    "Plan": "...",
    "Actions": [],
    "Evidence": [{"label": "answer-supporting claim", "value": "observed answer fact", "logical conflict": false, "citation": "timestamp/tool result", "evidence_status": "verified"}],
    "Finish": {
        "chain_complete": true,
        "completion_basis": "...",
        "answer": "..."
    }
}
Never output non-empty `Actions` together with a valid top-level `Finish`.

# Data Access Capabilities
Tools can access two tiers of video infomation. Specify your needs in the tool description:
- **Tier 1 (Pre-processed)**: Captions, Transcripts, and Knowledge Graphs, subject to each tool's schema.
- **Tier 2 (Raw Modal)**: Individual video frames (at specific timestamps) or raw video clips/segments.

# Tool Usage Boundaries
1. **Cross-tool verification**: By default, validate each evidence and the final conclusion with result from at least two different tools, especially for visual evidence, ambiguous evidence, fine-grained actions, future-event prediction, or when candidate answers are easily confused; Only if one tool already provides clear, direct, and decisive evidence with little ambiguity, you may finish without extra verification. Each evidence must undergo the above verification before it can be considered "evidence_status": verified.
2. **Web_Search is not local-video evidence**: The answer must come primarily from local video; do not arbitrarily use Web_Search to replace retrieval or frame checking, etc.
3. **Exact-value tasks need Evidence tracking**: When a query asks for exact values, totals, counts, prices, or scenario comparisons from the video, keep an `Evidence` array updated with one item per required fact/scenario.
4. **No fabricated tool history**: Never claim that search results were empty, a tool already failed, or prior attempts happened unless that information explicitly appears in the actual Observation history of this run.
5. **Image-timestamp alignment is explicit**: For both initial grounding collages and tool-returned collages, sub-frames are arranged in chronological order clockwise from the top-left: top-left, top-right, bottom-right, bottom-left. The timestamps provided alongside that collage correspond to those sub-frames in the same order.
6. **Attached images are already direct evidence**: If an Observation includes attached images from a visual tool, those images are already visible evidence for you. Do NOT call another raw image-returning tool on the same timestamps/window just to "inspect" the same images again. Re-probe only if you materially change the time window or apply `Crop`/`Zoom_in` to a returned image.
7. **Multiple-choice comparison**: Before Finish, compare the selected option against every visually plausible alternative using the same cited evidence. For a global character/property question, inspect distinct time regions and identify which appearance belongs to the main storyline. For a transition question such as "how did X leave", use a clip spanning before, during, and after the transition rather than a single frame.

# The Absolute Evidence Rule
1. **FORBIDDEN SIMULATION**: You are STRICTLY FORBIDDEN from guessing, using general knowledge, or simulating missing data (e.g., assuming a 2% payment rate).
2. **MISSING INFO = PROBE TRIGGER**: If a value is missing from the initial summary, it is a MANDATE to switch to tool use to probe evidence instead of guessing.

# Video-Centric Probe Hierarchy
1. **SCENARIO COMPLETENESS**: If a query asks for a comparison (Scenario A vs B) but only A is summarized, assume B is visible elsewhere. Use tools to probe the remainder of the timeline.
2. **VISUAL IS TRUTH**: Visual frames contain data that summaries often miss. If textual info is vague, your first priority is to define a visual analysis tool to "see" the exact data.
3. **LOGICAL RESTRAINT**: Logic tools (like `Python_Executor`) are ONLY for processing data that has already been retrieved from the evidence.
4. **TEMPORAL CONTINUITY**: After you find one promising segment, continue probing nearby windows before restarting a broad search.
5. **APPROXIMATE TEXT IS NOT FINAL**: If transcript/OCR says "around", "about", "approximately", "~", or similar, treat it as insufficient for final answering and re-check the exact screen region.
6. **CROSS-MODAL CHAIN IS PLANNER-DECLARED**: For exact-value video tasks, do not answer from a single modality. Use a chain like text->visual->answer or visual->text->answer. Only set `Finish.chain_complete=true` after you judge that the required chain is complete.
7. **PYTHON IS NOT A VISION FALLBACK**: Never use `Python_Executor` to infer what local video/images show from filenames or paths. For local video tasks, Python is allowed only for computation over structured evidence that tools have already extracted.

# Termination Protocol
- **MANDATORY CITATION**: Your final "answer" must explicitly cite the timestamp for every value used. If question contains options, "answer" must include a definite option.
- **CONDITIONAL ABORT**: You are FORBIDDEN from stating "information not present" until you have performed at least one visual/audio extraction action and received a null result.
- **STANDARD FINISH SIGNAL**: Use only the top-level `Finish` field. Do not emit `Finish` as a tool action inside `Actions`.
- **ROOT FINISH CONTRACT**: Planner must include top-level `Evidence` with a verified, cited value that directly states the selected answer fact, plus `Finish.chain_complete=true`, `completion_basis`, and `answer`. If the chain is incomplete, continue reasoning; do not pretend completion.
- **FAILURE STATEMENT**: Never repeat the exact call that returned `status=no_valid_result`. If the result reports a parameter or window validation error, correct those arguments and retry the tool on a materially valid scope; otherwise switch tools. Only if all relevant probes return no valid data, state: "The information [X] was not found after visual or audio probing of the video."
"""
# Subagent prompt
RECURSIVE_TASK_PROMPT = """You are a subagent for a video understanding system. Your responsibility is to transform an abstract tool request into executable tool calls and return the result.

# Input
- Tool: {function_name}
- Description: {abstract_tool_description}
- Parameters: {function_args}
- Goal: {goal}
- Parent Context: {parent_context}

# Runtime Control Tools
{tools_description}

# Role Boundaries
- Given an abstract tool request plus task context. You must not assume direct access to captions, transcripts, graph entries, or raw grounding blobs outside executable tool parameters/results.
- Tool search is automatic engine behavior, not a callable tool for you. Do NOT output `Tool_Search` in `Actions`.
- If the request is abstract or not directly executable, decide how it should be rewritten or decomposed; the engine will automatically run tool search and validation for the candidate tool names you output.
- Tool-search candidate lists are routing metadata for the sub-agent. Return executable actions or L4 child actions, not raw routing lists as final evidence.
- When the runtime provides `[Target Tool Schema]`, `[Candidate Tool Schemas]`, or `[Known Tool Schemas]`, treat their `parameters.properties` as the executable parameter contract. Use those field names directly instead of inventing aliases.

# Fixed Execution Priority
Follow this priority exactly and do not skip levels:
L1. **Tool exists and parameters are valid -> execute directly**
L2. **Tool exists but parameters are invalid/incomplete -> rewrite parameters for that same tool first**
L3. **Tool does not exist / is abstract -> choose one similar executable tool via route metadata and fill its parameters**
L4. **Single tool is insufficient -> decompose into multiple child tools and return them together**

# Decomposition Protocol
1. **Read Mediation Markers First**: Respect `[Mediation Rule]`, `[Allowed Resolution Modes]`, `[Locked Target Tool]`, `[Target Tool Schema]`, `[Candidate Tool Schemas]`, and `[Route Plan Candidates]`.
   If the runtime provides `[Composite Intent Guidance]` with `recommended_resolution="parallel_decomposition"`, treat that as a strong hint to prefer L4 over premature single-tool collapse.
2. If mediation is L2, keep the target tool fixed by default. Rewrite aliases, remove unsupported fields, repair time-window expressions, and keep only schema-compatible parameters.
   Treat tool-search schemas as authoritative parameter requirements. If the schema says `t_start`, `t_end`, and `density`, do not emit old aliases like `time_range_start`, `time_range_end`, `frame_count`, or placeholder identifiers like `video_id="local_video"`.
   If the locked target tool is `Python_Executor` but the planner is asking to search/find/inspect local video evidence, treat it as semantically incompatible and switch to retrieval/visual tools instead of wrapping the request in dummy Python.
3. **Escalate Only When Necessary**: In locked-target mode, you may switch to a replacement tool (L3) or parallel decomposition (L4) only if you judge the target tool is semantically incompatible or cannot complete the task alone.
4. If mediation is `L3 or L4`, use route metadata and candidate tools to decide between:
   - one executable replacement tool, or
   - multiple child tools returned together in `Actions`
   When the request simultaneously asks about multiple evidence dimensions such as actions, style/category, subtitles or audio cues, and scene distinctions/comparisons, prefer multiple concrete tools in one parallel L4 batch unless one single tool is clearly decisive and sufficient.
5. **Abstract-tool boundary is level-specific**:
   - In `L1-L3`, every returned action must already be a concrete executable tool. Do not output placeholder or abstract tool names.
   - Only in `L4` may you selectively return abstract child tools in `Actions`, when one tool is insufficient and the task must be decomposed for resolution.
6. **L4 Semantics**: For each child tool produced at `L4`, the runtime will run Tool_Search plus parameter validation independently.
   Child tools that become valid concrete tools will execute directly.
7. **Tool_Search is Automatic**: Never output `Tool_Search` as an action. Choose tool names and fill parameters; the engine will automatically search, validate, and execute.
8. **Multimodal Tool Results**: Tool observations may include structured JSON plus attached frame images. Use the textual `result` field for the normalized summary, but when images are attached you MUST inspect those images directly instead of assuming the summary is exhaustive. Those attached images are already direct evidence; do not chain another raw image-returning tool over the same timestamps unless you materially change the window or need `Crop`/`Zoom_in`.
   For both initial grounding collages and tool-returned collages, sub-frames are arranged in chronological order clockwise from the top-left: top-left, top-right, bottom-right, bottom-left, and the provided timestamps correspond to those sub-frames in the same order.
9. **Failure Handling**: Failed tool executions may be returned to you for another step. In that case, repair arguments, choose a better candidate tool, or change the probe range/query. But if a tool returned `status=no_valid_result`, treat that tool as invalid for the current branch and switch tools instead of retrying it with adjusted parameters.
10. **JSON-Only Discipline**: Do not narrate candidate-tool analysis, route comparison, or schema discussion outside the final JSON object. Keep `Thought` to one short sentence and `Plan` to 2 short sentences.
   Start the very first character of your reply with `{{` and end with `}}`.

# Operational Rules
1. Focus strictly on this function and task.
2. Do NOT output any prose outside the final JSON object.
3. **No Raw Data Access**: Access video/audio information by passing tool-compatible parameters or prior runtime result pointers (e.g., "$last_inspect_frame_result", "$frame_evidence_from_Inspect_Frame") into tool parameters.
4. **Evidence Priority**: Prioritize extracting evidences from the original video content using visual/audio tools, rather than relying on general knowledge.
5. **Schema-First Parameters**: When tool-search results include parameter schemas, copy the allowed parameter names from those schemas exactly. Unsupported keys are a bug, not a creative choice.

# Required Output JSON Schema
{{
    "Thought": "reasoning (max 2 short sentences)",
    "Plan": "steps",
    "Actions": [
        {{
            "tool": "ACTUAL_TOOL_NAME_HERE",
            "description": "Precise capability goal for the executable tool selected at runtime.",
            "params": {{"arg": "value"}}
        }}
    ],
}}
You may return one concrete tool action or multiple concrete tool actions.
Return exactly one JSON object and nothing else.
"""

# Few-shot examples split by query type

FEW_SHOT_TEXT_ONLY = """
# Few-Shot Examples (text / math / knowledge queries)

## Example 1: Direct computation (Root Planner)
User Query: "What is the compound interest on $5000 at 8% for 3 years?"
<think>
I need to calculate compound interest using the formula A = P(1 + r)^t. 
I will use Python_Executor for precision.
</think>
{"Thought":"This is a math problem involving compound interest.","Plan":"Compute the value using Python_Executor and finish.","Actions":[{"tool":"Python_Executor","params":{"code_script":"P=5000; r=0.08; t=3; ci=P*((1+r)**t - 1); print(f'Compound interest: {ci:.2f}')}}]}

Observation: [{"tool": "Python_Executor", "status": "success", "result": "Compound interest: 1258.56"}]

<think>
The calculation is complete. I will now provide the final answer.
</think>
{"Thought":"I have the calculated value.","Plan":"Return final response to the user.","Actions":[],"Finish":{"chain_complete":true,"completion_basis":"compute->answer","answer":"The total compound interest paid over 3 years is $1258.56."}}
"""

FEW_SHOT_VIDEO = """
# Video Planning Notes
- Prefer capability-first plans and let the runtime/sub-agent resolve or repair executable tools.
- Only emit a direct concrete action when it is already well grounded by the task and recent observations.
- Use parallel actions only when multiple evidence probes are independent and all parameters are grounded.
- For exact-value questions, keep `Evidence` updated and do not finish from approximate text alone.
"""


def get_runtime_control_tools_description(*, mediation_mode: bool = False) -> str:
    lines = [
        "- Tool search is automatic engine behavior. You do not call `Tool_Search`; instead, output candidate tool names plus parameters and the engine will search/validate automatically.",
        "- Python_Executor(code_script: str) for math/aggregation over extracted evidence only; never for inferring image/video content from file paths",
    ]
    if bool(settings.web_search_enabled):
        lines.append("- Web_Search(query: str)")
    lines.append(
        "- Sub-agents must output only executable Actions. Do NOT output Finish. A top-level `Result` note is optional and never replaces executable Actions."
    )
    lines.append(
        "- Successful tool results return directly to the parent planner. Only failed executions may be surfaced back to the sub-agent for retry/repair."
    )
    return "\n".join(lines)


def _escape_braces(text: str) -> str:
    """Escape braces to avoid str.format KeyError when descriptions contain JSON."""
    return text.replace("{", "{{").replace("}", "}}")


def _classify_query(query: str) -> str:
    """Heuristic classification of query type.

    Returns 'video' if the query likely relates to video/multimedia content,
    otherwise 'text'. The check is intentionally conservative: only explicit
    video keywords trigger the video branch so that text/math queries never
    receive video-specific few-shot examples.
    """
    if not query:
        return "text"
    q = query.lower()
    video_keywords = [
        "video", "clip", "scene", "frame", "footage",
        "watch", "camera", "visual", "subtitle", "transcript",
        "mlvu task type", "mvbench task type", "videomme",
        "\u64ad\u653e", "\u89c6\u9891", "\u753b\u9762", "\u955c\u5934", "\u7247\u6bb5",
    ]
    return "video" if any(kw in q for kw in video_keywords) else "text"


def query_requires_evidence_first(query: str) -> bool:
    """Heuristic for exact-value local-evidence tasks."""
    if not query:
        return False
    q = query.lower()
    benchmark_markers = ["mlvu task type", "mvbench task type", "videomme"]
    if any(marker in q for marker in benchmark_markers):
        return True
    exact_markers = [
        "exact", "exactly", "precise", "specific", "quoted",
        "total", "amount", "value", "values", "number", "numbers",
        "how much", "how many", "price", "prices", "figure", "figures",
    ]
    comparison_markers = [
        "compare", "comparison", "versus", "vs", "scenario", "scenarios",
        "both", "each", "respectively", "difference", "baseline", "upgraded",
    ]
    evidence_markers = [
        "video", "clip", "screen", "frame", "shown", "showing", "visible",
        "subtitle", "transcript", "caption", "scene",
    ]
    has_exact = any(marker in q for marker in exact_markers)
    has_comparison = any(marker in q for marker in comparison_markers)
    has_video = any(marker in q for marker in evidence_markers)
    return has_exact and (has_comparison or has_video)


def estimate_required_evidence_slots(query: str) -> int:
    """Infer the minimum number of evidence slots needed for finish gating."""
    if not query_requires_evidence_first(query):
        return 0
    q = query.lower()
    multi_markers = ("versus", " vs ", "compare", "comparison", "both", "each", "respectively", "scenario")
    return 2 if any(marker in q for marker in multi_markers) else 1


def infer_mvbench_route_prior(query: str) -> str:
    if not query:
        return ""
    text = " ".join(str(query).lower().split())
    task_type = ""
    match = re.search(r"task\s*type\s*:\s*([^\n]+)", str(query), flags=re.IGNORECASE)
    if match:
        task_type = str(match.group(1)).strip().lower()

    lines = []
    if "object existence" in task_type or any(token in text for token in ("is there", "are there", "present", "visible", "appear")):
        lines.append("- Route prior: Object Existence -> prefer direct visual confirmation before computation or broad retrieval.")
    if "action localization" in task_type or any(token in text for token in ("before", "after", "when", "locate", "which segment")):
        lines.append("- Route prior: Action Localization -> refine the time window first, then verify with local clip/frame evidence.")
    if any(token in task_type for token in ("action count", "moving count")) or any(token in text for token in ("how many", "count", "number of times")):
        lines.append("- Route prior: Action/Moving Count -> collect grounded occurrences first, then use aggregation/meta tools to merge and count.")
    if "egocentric navigation" in task_type or any(token in text for token in ("camera wearer", "egocentric", "turn left", "turn right", "go straight", "navigation")):
        lines.append("- Route prior: Egocentric Navigation -> prefer continuous visual evidence or clip-level video inspection over isolated OCR probes.")
    if "task type:" in text and "options:" in text and "exactly one best option" in text:
        lines.append(
            "- Route prior: MVBench MCQ -> for short or already-bounded clips, prefer one decisive clip-level verification "
            "before repeating same-window frame probes."
        )
    return "\n".join(lines)


def get_system_prompt(
    tools_description: str = "",
    include_examples: bool = True,
    query: str = "",
    grounding_context: str = "",
) -> str:
    """Get the root planner system prompt.

    Args:
        tools_description: Ignored for root planner (kept for backward compat).
        include_examples: Whether to include few-shot examples
        query: The user query (used to pick domain-appropriate examples)
        grounding_context: Query-retrieved grounding evidence (metadata/caption/asr/ocr)

    Returns:
        Complete system prompt
    """
    prompt = SYSTEM_PROMPT
    evidence_first = query_requires_evidence_first(query)
    route_prior = infer_mvbench_route_prior(query)
    if grounding_context:
        prompt += (
            "\n\n# Grounding Context (query + top-k evidence)\n"
            "Use this context as first-class evidence and do not ignore it. "
            "If evidence is insufficient or conflicting, call retrieval tools to gather more.\n"
            f"{grounding_context}"
        )
    if not bool(settings.web_search_enabled):
        prompt += "\nExternal web search is disabled in this runtime."
    if route_prior:
        prompt += (
            "\n\n# Route Prior\n"
            "Use these task-level routing priors as soft guidance when choosing the next capability:\n"
            f"{route_prior}"
        )
    if evidence_first:
        prompt += (
            "\n\n# Evidence-First Capability Guide\n"
            f"{EVIDENCE_FIRST_CAPABILITY_GUIDE}\n"
            "Describe the missing capability precisely. The runtime and sub-agent will route it to concrete executable tools."
        )
        prompt += (
            "\n\n# Evidence-First Finish Policy\n"
            f"- This query requires at least {estimate_required_evidence_slots(query)} evidence slot(s).\n"
            "- Keep `Evidence` updated until each required item has a non-empty `value`, a citation, and "
            "`evidence_status=\"exact\"`.\n"
            "- At Finish, at least one verified Evidence label/value must explicitly state the selected answer fact.\n"
            "- Use both text/retrieval and direct visual probes when both modalities contain relevant evidence. "
            "For a purely visual event, decisive direct visual evidence may complete the chain without fabricated text support.\n"
            "- For fine-grained action tasks such as needle retrieval, a Video_Clip_QA result must directly state the selected action.\n"
            "- Do not finish with approximate evidence.\n"
            "- Do not call math or web tools as a substitute for local video evidence.\n"
            "- The system validates the `Finish` contract and evidence thresholds. If rejected, follow the "
            "verifier feedback and acquire or reconcile the missing evidence."
        )
        if "mlvu task type: needle" in str(query or "").lower():
            prompt += (
                "\n- Needle localization rule: first identify the initial grounding segment that contains the "
                "question entities, then call Video_Clip_QA on that exact short window. Do not scan the full "
                "video or jump to distant semantic-retrieval hits while the entity-matching grounding window "
                "has not been checked dynamically."
            )
        if "mlvu task type: anomaly_reco" in str(query or "").lower():
            prompt += (
                "\n- Anomaly decision rule: after a focused Video_Clip_QA directly describes theft, shooting, "
                "assault, or a road accident, map that behavior to the listed option and Finish with its citation. "
                "Do not repeat the same frame window to seek generic confirmation after decisive anomaly evidence. "
                "Never map an inconclusive visual answer or a semantic-retrieval rank to an anomaly option; instead "
                "inspect a different focused window until the observed behavior is explicit."
            )
        if "mlvu task type: count" in str(query or "").lower():
            prompt += (
                "\n- Count rule: count contiguous video scenes that match the requested action, using the grounded "
                "caption/transcript and a focused visual check. Do not reject a captioned action merely because motion "
                "is subtle in sampled frames. If exactly one localized segment matches, select the option for 1."
            )
        if "mlvu task type: plotqa" in str(query or "").lower():
            prompt += (
                "\n- Plot rule: when an entity-matching grounding segment explicitly names an answer entity in spoken "
                "dialogue or captions, treat that direct statement as decisive and map it to the matching option; do not "
                "replace it with speculative character identification from unrelated frames."
            )
    if include_examples:
        qtype = _classify_query(query)
        # For text/math queries, examples can over-anchor small local models and
        # cause template-answer copying. Keep examples for video/multimodal only.
        if qtype == "video":
            prompt += "\n\n" + FEW_SHOT_VIDEO
    return prompt


def get_recursive_prompt(
    function_name: str,
    function_args: str,
    goal: str,
    parent_context: str,
    tools_description: str,
    abstract_tool_description: str = "",
) -> str:
    """Get prompt for recursive sub-agent.

    Args:
        function_name: Name of function to implement
        function_args: Arguments passed to function
        goal: Goal description
        parent_context: Context from parent agent
        tools_description: Available tools
        abstract_tool_description: Planner's description of what the
            abstract tool is supposed to do (semantic intent).

    Returns:
        Recursive task prompt
    """
    safe_desc = _escape_braces(tools_description)
    safe_abstract = _escape_braces(abstract_tool_description) if abstract_tool_description else "No description provided by planner."
    return RECURSIVE_TASK_PROMPT.format(
        function_name=function_name,
        function_args=function_args,
        goal=goal,
        parent_context=parent_context,
        tools_description=safe_desc,
        abstract_tool_description=safe_abstract,
    )
